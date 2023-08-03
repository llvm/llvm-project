#include <iostream>

int returnsInt()  {
  return 42;
}

int returnsIntUnused() {
  return 37;
}

void throwException() {
  throw "Division by zero condition!";
}

struct CxxClass {
  int a = 100;
  int b = 101;

  int sum() {
    return a + b;
  }
};

struct CxxSubclass: public CxxClass {
  int c = 102;
};



