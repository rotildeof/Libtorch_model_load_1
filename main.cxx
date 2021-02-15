#include <iostream>
#include <string>
#include <torch/torch.h>

struct NetImpl : torch::nn::Module {
    NetImpl() {
        fc1 = register_module("fc1", torch::nn::Linear(2, 4));
        fc2 = register_module("fc2", torch::nn::Linear(4, 4));
        fc3 = register_module("fc3", torch::nn::Linear(4, 2));
    }
    torch::nn::Linear fc1 = {nullptr};
    torch::nn::Linear fc2 = {nullptr};
    torch::nn::Linear fc3 = {nullptr};
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = torch::softmax(fc3->forward(x), 1);
        return x;
    }
};

TORCH_MODULE(Net);

int main() {
    Net net;
    torch::load(net, "../model.pt");
    std::vector<float> v{0, 0, 0, 1, 1, 0, 1, 1};
    torch::Tensor tensor = torch::from_blob(v.data(), {4, 2});

    std::cout << net->forward(tensor) << std::endl;
    return 0;
}