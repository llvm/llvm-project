#include <iostream>
#include <unistd.h>

void foo() {
    std::cout << "This is foo" << std::endl;
}

int main() {
    std::cout << "hello world" << std::endl;
    foo();
    std::cout << "after foo" << std::endl;
    return 0;
}
