// RUN: %libomp-cxx20-compile-and-run | FileCheck %s --match-full-lines

#ifndef HEADER
#define HEADER

#include <cstdlib>
#include <cstdarg>
#include <cstdio>
#include <vector>

struct Reporter {
  const char *name;

  Reporter(const char *name) : name(name) { print("ctor"); }

  Reporter() : name("<anon>") { print("ctor"); }

  Reporter(const Reporter &that) : name(that.name) { print("copy ctor"); }

  Reporter(Reporter &&that) : name(that.name) { print("move ctor"); }

  ~Reporter() { print("dtor"); }

  const Reporter &operator=(const Reporter &that) {
    print("copy assign");
    this->name = that.name;
    return *this;
  }

  const Reporter &operator=(Reporter &&that) {
    print("move assign");
    this->name = that.name;
    return *this;
  }

  struct Iterator {
    const Reporter *owner;
    int pos;

    Iterator(const Reporter *owner, int pos) : owner(owner), pos(pos) {}

    Iterator(const Iterator &that) : owner(that.owner), pos(that.pos) {
      owner->print("iterator copy ctor");
    }

    Iterator(Iterator &&that) : owner(that.owner), pos(that.pos) {
      owner->print("iterator move ctor");
    }

    ~Iterator() { owner->print("iterator dtor"); }

    const Iterator &operator=(const Iterator &that) {
      owner->print("iterator copy assign");
      this->owner = that.owner;
      this->pos = that.pos;
      return *this;
    }

    const Iterator &operator=(Iterator &&that) {
      owner->print("iterator move assign");
      this->owner = that.owner;
      this->pos = that.pos;
      return *this;
    }

    bool operator==(const Iterator &that) const {
      owner->print("iterator %d == %d", 2 - this->pos, 2 - that.pos);
      return this->pos == that.pos;
    }

    bool operator!=(const Iterator &that) const {
      owner->print("iterator %d != %d", 2 - this->pos, 2 - that.pos);
      return this->pos != that.pos;
    }

    Iterator &operator++() {
      owner->print("iterator prefix ++");
      pos -= 1;
      return *this;
    }

    Iterator operator++(int) {
      owner->print("iterator postfix ++");
      auto result = *this;
      pos -= 1;
      return result;
    }

    int operator*() const {
      int result = 2 - pos;
      owner->print("iterator deref: %i", result);
      return result;
    }

    size_t operator-(const Iterator &that) const {
      int result = (2 - this->pos) - (2 - that.pos);
      owner->print("iterator distance: %d", result);
      return result;
    }

    Iterator operator+(int steps) const {
      owner->print("iterator advance: %i += %i", 2 - this->pos, steps);
      return Iterator(owner, pos - steps);
    }
  };

  Iterator begin() const {
    print("begin()");
    return Iterator(this, 2);
  }

  Iterator end() const {
    print("end()");
    return Iterator(this, -1);
  }

  void print(const char *msg, ...) const {
    va_list args;
    va_start(args, msg);
    printf("[%s] ", name);
    vprintf(msg, args);
    printf("\n");
    va_end(args);
  }
};

int main() {
  printf("do\n");
  Reporter C("C");
  Reporter D("D");
#pragma omp fuse
  {
    for (auto it = C.begin(); it != C.end(); ++it)
      printf("v=%d\n", *it);

    for (auto it = D.begin(); it != D.end(); ++it)
      printf("vv=%d\n", *it);
  }
  printf("done\n");
  return EXIT_SUCCESS;
}

#endif /* HEADER */

// CHECK: do
// CHECK: [C] ctor
// CHECK-NEXT: [D] ctor
// CHECK-NEXT: [C] begin()
// CHECK-NEXT: [C] begin()
// CHECK-NEXT: [C] end()
// CHECK-NEXT: [C] iterator distance: 3
// CHECK-NEXT: [D] begin()
// CHECK-NEXT: [D] begin()
// CHECK-NEXT: [D] end()
// CHECK-NEXT: [D] iterator distance: 3
// CHECK-NEXT: [C] iterator advance: 0 += 0
// CHECK-NEXT: [C] iterator move assign
// CHECK-NEXT: [C] iterator deref: 0
// CHECK-NEXT: v=0
// CHECK-NEXT: [C] iterator dtor
// CHECK-NEXT: [D] iterator advance: 0 += 0
// CHECK-NEXT: [D] iterator move assign
// CHECK-NEXT: [D] iterator deref: 0
// CHECK-NEXT: vv=0
// CHECK-NEXT: [D] iterator dtor
// CHECK-NEXT: [C] iterator advance: 0 += 1
// CHECK-NEXT: [C] iterator move assign
// CHECK-NEXT: [C] iterator deref: 1
// CHECK-NEXT: v=1
// CHECK-NEXT: [C] iterator dtor
// CHECK-NEXT: [D] iterator advance: 0 += 1
// CHECK-NEXT: [D] iterator move assign
// CHECK-NEXT: [D] iterator deref: 1
// CHECK-NEXT: vv=1
// CHECK-NEXT: [D] iterator dtor
// CHECK-NEXT: [C] iterator advance: 0 += 2
// CHECK-NEXT: [C] iterator move assign
// CHECK-NEXT: [C] iterator deref: 2
// CHECK-NEXT: v=2
// CHECK-NEXT: [C] iterator dtor
// CHECK-NEXT: [D] iterator advance: 0 += 2
// CHECK-NEXT: [D] iterator move assign
// CHECK-NEXT: [D] iterator deref: 2
// CHECK-NEXT: vv=2
// CHECK-NEXT: [D] iterator dtor
// CHECK-NEXT: [D] iterator dtor
// CHECK-NEXT: [D] iterator dtor
// CHECK-NEXT: [C] iterator dtor
// CHECK-NEXT: [C] iterator dtor
// CHECK-NEXT: done
// CHECK-NEXT: [D] iterator dtor
// CHECK-NEXT: [C] iterator dtor
// CHECK-NEXT: [D] dtor
// CHECK-NEXT: [C] dtor
