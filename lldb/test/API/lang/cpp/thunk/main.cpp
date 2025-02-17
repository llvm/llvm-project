#include <stdio.h>

class Base1 {
public:
  virtual ~Base1() {}
};

class Base2 {
public:
  virtual void doit() = 0;
};

Base2 *b;

class Derived1 : public Base1, public Base2 {
public:
  virtual void doit() { printf("Derived1\n"); }
};

class Derived2 : public Base2 {
public:
  virtual void doit() { printf("Derived2\n"); }
};

void testit() { b->doit(); }

int main() {

  b = new Derived1();
  testit();

  b = new Derived2();
  testit();

  return 0;
}
