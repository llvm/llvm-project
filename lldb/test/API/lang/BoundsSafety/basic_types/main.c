#include <ptrcheck.h>

void *store;
void consume(void* p) {
  store = p;
}

struct A { int x; int y;};

int f1(int *__bidi_indexable arr) { return *arr; }

int f(char *__indexable i_arr, char *__bidi_indexable bi_arr,
       int *__indexable i_arri, int *__bidi_indexable bi_arri,
       float *__indexable i_arrf, float *__bidi_indexable bi_arrf,
       double *__indexable i_arrd, double *__bidi_indexable bi_arrd,
       struct A *__indexable i_arrA, struct A *__bidi_indexable bi_arrA) {
    // Calls to consume are to ensure the `__bidi_indexable` parameters
    // have debug info emitted (rdar://102116986).
    // We break here to workaround a bug where LLDB is unable to print
    // the value of `bi_arr` after its last use (rdar://102149316).
    consume(bi_arr); // break here
    consume(bi_arri);
    consume(bi_arrf);
    consume(bi_arrd);
    consume(bi_arrA);

    return i_arri[0];
}

int main() {
    char arr[] = "hello world";
    int arri[] = {1,2,3};
    float arrf[] = {1.0f, 2.0f, 3.0f};
    double arrd[] = {1.0, 2.0, 3.0};
    struct A arrA[] = {{1,2}, {3,4}, {4,5}};
    int (*callback)(int *__bidi_indexable) = f1;

    f(arr, arr,
      arri, arri,
      arrf, arrf,
      arrd, arrd,
      arrA, arrA);

    int (*arrp)[8];
    int arre[8] = {1,2,3,4,5,6,7,8};
    arrp = &arre;

    return 0; // break here
}
