#include <stdio.h>

class Base1 {
public:
  virtual ~Base1() {}
};

class Base2 {
public:
  virtual void doit() = 0;
  virtual void doit_debug() = 0;
};

Base2 *b;

class Derived1 : public Base1, public Base2 {
public:
  virtual void doit() { printf("Derived1\n"); }
  virtual void __attribute__((nodebug)) doit_debug() {
    printf("Derived1 (no debug)\n");
  }
};

class Derived2 : public Base2 {
public:
  virtual void doit() { printf("Derived2\n"); }
  virtual void doit_debug() { printf("Derived2 (debug)\n"); }
};

void testit() { b->doit(); }

void testit_debug() {
  b->doit_debug();
  printf("This is where I should step out to with nodebug.\n"); // Step here
}

int main() {

  b = new Derived1();
  testit();
  testit_debug();

  b = new Derived2();
  testit();
  testit_debug();

  return 0;
}
