#include <stdio.h>

struct Foo {
  Foo(int _n) : n(_n) {}

  int n = 0;
  struct {
    struct Foo *stqe_next;
  } next = {};
};

struct foos {
  struct Foo *stqh_first;
  struct Foo **stqh_last;
} foos = {nullptr, &(foos).stqh_first};

int main() {
  bool removed;
  do {
    removed = false;
    for (Foo *fi = foos.stqh_first; fi; fi = fi->next.stqe_next) {
      printf("%p\n", fi); // False UAF reported here
      if (fi->n == 1) {
        // STAILQ_REMOVE expanded - the if check below is key to the bug
        if (foos.stqh_first == fi) {
          // STAILQ_REMOVE_HEAD
          foos.stqh_first = foos.stqh_first->next.stqe_next;
          if (foos.stqh_first == nullptr)
            foos.stqh_last = &foos.stqh_first;
        } else {
          // Remove from middle/end
          Foo *curelm = foos.stqh_first;
          while (curelm->next.stqe_next != fi)
            curelm = curelm->next.stqe_next;
          if ((curelm->next.stqe_next =
                   curelm->next.stqe_next->next.stqe_next) == nullptr)
            foos.stqh_last = &curelm->next.stqe_next;
        }
        delete fi;
        removed = true;
        break;
      }
    }
  } while (removed);
  return 0;
}
