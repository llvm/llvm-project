#include <ptrcheck.h>

typedef struct {
    int len;
    int *__counted_by(len) buf;
} S;

char foo(char *__sized_by(l) strArg, unsigned int l) {
    return strArg[l-1]; // break here
}

int main() {
    int arr[] = {1, 2, 3, 4, 5};
    int len;
    int *__counted_by(len) buf;
    len = sizeof(arr) / sizeof(int);
    buf = arr;

    S s = {len, buf};

    char *str = "Hello!";
    char ch = foo(str, 6);
    return 0; // break here
}
