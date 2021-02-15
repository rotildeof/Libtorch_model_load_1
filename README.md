# Libtorch_model_load_1
Example of loading pre-trained model with libtorch.  
モデルの読み込み方法。ここでは、Libtorch_example_2 (https://github.com/rotildeof/Libtorch_example_2 ) で学習したモデルを読み込んで使う方法を述べる。

前準備
-

```c++
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

```

前回保存した`model.pt`を読み込むには少し工夫がいる。まず、学習時のニューラルネットワークの設計部分をそのまま持ってくる。ただし、このとき構造体の名前を"構造体名"+"Impl"としなければならない(上の例ではNetImpl)。このように名前を付けるのは、そのあとのマクロ`TORCH_MODULE(Net)`によってクラス`Net`を新しく自動生成させるためである(https://recruit.cct-inc.co.jp/tecblog/deep-learning/pytorch-cpp_02/ にわかりやすい説明が書かれてある)。何故かはわからないが、こうしないとモデルを読み込むことができない。

モデル読み込み
-
main 関数内の
```c++
    Net net;
    torch::load(net, "../model.pt");
```
の部分でモデルを読み込んでいる。
上のコードのように、まず`Net` のインスタンスを作り、`torch::load()`の第１引数に渡す。第２引数は保存した`model.pt`までのパスを渡す。
あとは学習時と同じように`torch::Tensor`型の入力を作り、`net->forward()`で推論の結果を得ることができる。

出力
-
```
 0.0001  0.9999
 0.9999  0.0001
 0.9999  0.0001
 0.0000  1.0000
[ CPUFloatType{4,2} ]
```

出力結果の取得法
-

```c++
auto output = net->forward(tensor);
```
などで推論結果を得たあと、具体的な数値を取得するには
```c++
auto a = output[0][0].item<float>();
auto b = output[0][1].item<float>();
```
などとすればよい。または、
```c++
auto c = output[0].data_ptr<float>();
```
で、出力配列の先頭ポインタを取得できる。上の例では、０番目のデータの出力配列の先頭ポインタを取得している。あとは、`c[0]`, `c[1]` などで要素にアクセスできる。