#include <cstdio>

class Base {
public:
  virtual int value() { return 10; }
  virtual ~Base() = default;
};

class Derived : public Base {
public:
  int value() override { return 20; }
};

class OtherDerived : public Base {
public:
  int value() override { return 30; }
};

int call_value(Base *obj) { return obj->value(); }

int main() {
  Derived d;
  OtherDerived od;
  Base *base_ptr = &d;
  printf("%d %d %d\n", d.value(), od.value(), base_ptr->value());
  return 0; // break here
}
