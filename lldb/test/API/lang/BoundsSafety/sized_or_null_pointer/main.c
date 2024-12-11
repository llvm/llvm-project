#include <ptrcheck.h>

typedef struct {
    int len;
    int *__counted_by_or_null(len) buf;
} S;

char foo(char *__sized_by_or_null(l) strArg, unsigned int l) {
    return strArg[l-1]; // break here
}

int main() {
    int arr[] = {1, 2, 3, 4, 5};
    int len;
    int *__counted_by_or_null(len) buf;
    len = sizeof(arr) / sizeof(int);
    buf = arr;

    S s = {len, buf};

    char *str = "Hello!";
    char ch = foo(str, 6);
    // break here
    char ch2 = foo((void*)0, 6);
    return 0;
}
