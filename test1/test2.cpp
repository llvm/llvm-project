#include <stdio.h>
#include <sys/queue.h>

struct Foo {
  Foo(int _n) : n(_n) {}

  int n = 0;
  STAILQ_ENTRY(Foo) next = {};
};

STAILQ_HEAD(foos, Foo)
foos = STAILQ_HEAD_INITIALIZER(foos);

int main() {
  Foo *fi;

  for (int i = 1; i <= 3; i++) {
    fi = new Foo(i);
    STAILQ_INSERT_TAIL(&foos, fi, next);
  }

  STAILQ_FOREACH(fi, &foos, next) {
    printf("%d\n", fi->n);
  }
  printf("\n");

  bool removed = false;
  do {
    removed = false;
    printf("start\n");
    STAILQ_FOREACH(fi, &foos, next) {
      printf("%p\n", fi);  // (1) False UAF reported here
      if (fi->n == 1) {
        STAILQ_REMOVE(&foos, fi, Foo, next);
        printf(" delete %p\n", fi);
        delete fi;
        // fi->n = 2; // (2) THIS is UAF
        removed = true;
        break;
      }
    }
  } while (removed);
  printf("\n");

  STAILQ_FOREACH(fi, &foos, next) {
    printf("%d\n", fi->n);
  }

  return 0;
}
