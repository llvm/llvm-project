#include <stdio.h>

void foo(int arg) {
    printf("%d\n", arg);
}

int main() {
    int x = 10;
    foo(x);
    return 0;
}
