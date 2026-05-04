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

    void print(const char *msg) const { owner->print(msg); }
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
#pragma omp unroll partial(2)
  for (Reporter c{"init-stmt"}; auto &&v : Reporter("range"))
    printf("v=%d\n", v);
  printf("done\n");
  return EXIT_SUCCESS;
}

#endif /* HEADER */

// CHECK:      do
// CHECK-NEXT: [init-stmt] ctor
// CHECK-NEXT: [range] ctor
// CHECK-NEXT: [range] begin()
// CHECK-NEXT: [range] end()
// CHECK-NEXT: [range] iterator 0 != 3
// CHECK-NEXT: [range] iterator deref: 0
// CHECK-NEXT: v=0
// CHECK-NEXT: [range] iterator prefix ++
// CHECK-NEXT: [range] iterator 1 != 3
// CHECK-NEXT: [range] iterator deref: 1
// CHECK-NEXT: v=1
// CHECK-NEXT: [range] iterator prefix ++
// CHECK-NEXT: [range] iterator 2 != 3
// CHECK-NEXT: [range] iterator deref: 2
// CHECK-NEXT: v=2
// CHECK-NEXT: [range] iterator prefix ++
// CHECK-NEXT: [range] iterator 3 != 3
// CHECK-NEXT: [range] iterator dtor
// CHECK-NEXT: [range] iterator dtor
// CHECK-NEXT: [range] dtor
// CHECK-NEXT: [init-stmt] dtor
// CHECK-NEXT: done
