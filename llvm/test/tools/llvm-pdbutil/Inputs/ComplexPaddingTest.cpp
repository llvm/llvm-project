// Compile with "cl /c /Zi /GR- ComplexPaddingTest.cpp"
// Link with "link ComplexPaddingTest.obj /debug /nodefaultlib /entry:main"

#include <stdint.h>

extern "C" using at_exit_handler = void();

int atexit(at_exit_handler handler) { return 0; }

struct TestVB {
  static void operator delete(void *ptr, size_t sz) {}
  virtual ~TestVB() {}
  virtual void IntroFunction1() {}
  int X;
} A;

struct TestNVB {
  static void operator delete(void *ptr, size_t sz) {}
  virtual ~TestNVB() {}
  virtual void IntroFunction2() {}
  int Y;
} B;

struct TestVBLayout
    : public virtual TestVB,
      public TestNVB {
  static void operator delete(void *ptr, size_t sz) {}
  int Z;
} C;

struct TestIVBBase : public virtual TestVB {
  int A;
} D;

struct TestIVBDerived : public TestIVBBase {
  int B;
} E;

struct TestIVBMergedDerived
    : public virtual TestVB,
      public TestIVBBase {
  int B;
} F;

int main(int argc, char **argv) {

  return 0;
}
