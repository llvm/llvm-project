#include <ptrcheck.h>

int foo(int *__bidi_indexable biArg) {
    return *biArg;
}

int bar(int *sArg) {
    return *sArg;
}

int *__bidi_indexable baz(int *__bidi_indexable biArg) {
    return biArg;
}

int main() {
    int arr[] = {1, 2, 3, 4, 5};
    int *ptrImplicitBidiIndex = arr;
    int *__bidi_indexable ptrBidiIndex = arr + 1;
    int *__indexable ptrIndex = arr + 2;
    int *__single ptrSingle = arr + 3;
    int *__unsafe_indexable ptrUnsafe = arr + 4;
    int (*callback)(int *__bidi_indexable) = foo;
    int (*callback_s)(int *) = bar;
    callback(arr); // break here
}
