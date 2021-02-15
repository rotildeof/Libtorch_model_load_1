cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(load_test CXX)

list(APPEND CMAKE_PREFIX_PATH ~/Torch/libtorch)
find_package(Torch REQUIRED)

add_executable(load_test main.cxx)
add_compile_options(-Wall)
target_link_libraries(load_test "${TORCH_LIBRARIES}")
set_property(TARGET load_test PROPERTY CXX_STANDARD 14)
