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

    C& c_ref = c;
    C* c_ptr = &c;

    return; // Set a breakpoint here
  }

  void TestAddressOf(int param) {
    int x = 42;
    int& r = x;
    int* p = &x;
    int*& pr = p;

    typedef int*& mypr;
    mypr my_pr = p;

    std::string s = "hello";
    const char* s_str = s.c_str();

    char c = 1;

    return;
  }

 private:
  int field_ = 1;
};

int
main(int argc, char **argv)
{
  TestMethods tm;

  tm.TestInstanceVariables();
  return 0;
}
