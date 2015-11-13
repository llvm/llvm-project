// compile & generate coverage data using:
// clang++ -g -o test-linux_x86_64 -fsanitize=address -fsanitize-coverage=edge *.cpp
// ASAN_OPTIONS="coverage=1" ./test-linux_x86_64 && mv test-linux_x86_64.*.sancov test-linux_x86_64.sancov
// ASAN_OPTIONS="coverage=1" ./test-linux_x86_64 1 && mv test-linux_x86_64.*.sancov test-linux_x86_64-1.sancov

#include <stdio.h>
#include <string>

void foo();

__attribute__((noinline))
std::string bar(std::string str) { printf("bar\n"); return str; }

int main(int argc, char **argv) {
    if (argc == 2)
        foo();
    bar("str");
    printf("main\n");
}
