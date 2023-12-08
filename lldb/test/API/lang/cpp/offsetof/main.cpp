#include <cstdint>

class Base {
  int32_t a;
};
class Class1 : Base {
public:
  int32_t b;
};

class EmptyBase {
};
class Class2 : EmptyBase {
public:
  int32_t b;
};

int main(int argc, char **argv) {
  Class1 c1;
  Class2 c2;
  //% self.expect("expr offsetof(Base, a)", substrs=["= 0"])
  //% self.expect("expr offsetof(Class1, b)", substrs=["= 4"])
  //% self.expect("expr offsetof(Class2, b)", substrs=["= 0"])
  return c1.b + c2.b;
}
