// RUN: %libomp-cxx20-compile-and-run | FileCheck %s --match-full-lines

#include <cstdlib>
#include <cstdarg>
#include <cstdio>

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
      owner->print("iterator %d == %d", this->pos, that.pos);
      return this->pos == that.pos;
    }

    bool operator!=(const Iterator &that) const {
      owner->print("iterator %d != %d", this->pos, that.pos);
      return this->pos != that.pos;
    }

    Iterator &operator++() {
      owner->print("iterator prefix ++");
      pos += 1;
      return *this;
    }

    Iterator operator++(int) {
      owner->print("iterator postfix ++");
      auto result = *this;
      pos += 1;
      return result;
    }

    int operator*() const {
      owner->print("iterator deref: %d", pos);
      return pos;
    }

    size_t operator-(const Iterator &that) const {
      int result = this->pos - that.pos;
      owner->print("iterator distance: %d", result);
      return result;
    }

    Iterator operator+(int steps) const {
      owner->print("iterator advance: %d += %d", this->pos, steps);
      return Iterator(owner, pos + steps);
    }
  };

  Iterator begin() const {
    print("begin()");
    return Iterator(this, 0);
  }

  Iterator end() const {
    print("end()");
    return Iterator(this, 4);
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
  Reporter range("range");
#pragma omp split counts(1, omp_fill, 1)
  for (auto it = range.begin(); it != range.end(); ++it)
    printf("v=%d\n", *it);
  printf("done\n");
  return EXIT_SUCCESS;
}

// CHECK: do
// CHECK: [range] ctor
// CHECK: v=0
// CHECK: v=1
// CHECK: v=2
// CHECK: v=3
// CHECK: done
// CHECK: [range] dtor
