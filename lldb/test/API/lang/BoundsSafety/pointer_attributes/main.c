#include <ptrcheck.h>

typedef struct {
    void *__sized_by(len) vptr;
    int len;
} S;

int main() {
    char arr[] = {1, 2, 3, 4};
    S a = {arr, sizeof(arr)};
    S *__bidi_indexable biPtr = &a;
    S *__indexable iPtr = &a;
    S *__single sPtr = &a;
    S *__unsafe_indexable uiPtr = &a;
    S *uPtr = &a;

    biPtr = &a;

    return 0; // break here
}
