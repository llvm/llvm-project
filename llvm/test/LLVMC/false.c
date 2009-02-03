// Test that we can compile .c files as C++ and vice versa
// RUN: llvmc -x c++ %s -x c %p/test_data/false.cpp -x lisp -x whatnot -x none %p/test_data/false2.cpp -o %t
// RUN: ./%t | grep hello

#include <iostream>

extern "C" void test();
extern std::string test2();

int main() {
    std::cout << "h";
    test();
    std::cout << test2() << '\n';
}
