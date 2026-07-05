#include <string>

class C {
public:
  int field_ = 1337;
};

class TestMethods {
public:
  void TestInstanceVariables() {
    C c;
    c.field_ = -1;

    return; // Set a breakpoint here
  }
};

int main(int argc, char **argv) {
  TestMethods tm;

  tm.TestInstanceVariables();
  return 0;
}
