// Compile with "cl /c /Zi /GR- SimplePaddingTest.cpp"
// Link with "link SimplePaddingTest.obj /debug /nodefaultlib /entry:main"

#include <stdint.h>

extern "C" using at_exit_handler = void();

int atexit(at_exit_handler handler) { return 0; }

struct SimplePadNoPadding {
  int32_t X;
  int32_t Y;
  // No padding anywhere, sizeof(T) = 8
} A;

struct SimplePadUnion {
  union {
    int32_t X;
    int64_t Y;
    struct {
      int32_t X;
      // 4 bytes of padding here
      int64_t Y;
    } Z;
  };
  // Since the padding occurs at a location that is occupied by other storage
  // (namely the Y member), the storage will still be considered used, and so
  // there will be no unused bytes in the larger class.  But in the debug
  // info for the nested struct, we should see padding.
  // sizeof(SimplePadUnion) == sizeof(Z) == 16
} B;

struct SimplePadNoPadding2 {
  bool A;
  bool B;
  bool C;
  bool D;
  // No padding anywhere, sizeof(T) = 4
} C;

struct alignas(4) SimplePadFields1 {
  char A;
  char B;
  char C;
  // 1 byte of padding here, sizeof(T) = 4
} E;

struct SimplePadFields2 {
  int32_t Y;
  char X;
} F;

struct SimplePadBase {
  // Make sure this class is 4 bytes, and the derived class requires 8 byte
  // alignment, so that padding is inserted between base and derived.
  int32_t X;
  // No padding here
} G;

struct SimplePadDerived : public SimplePadBase {
  // 4 bytes of padding here due to Y requiring 8 byte alignment.
  // Thus, sizeof(T) = 16
  int64_t Y;
} H;

struct SimplePadEmptyBase1 {};
struct SimplePadEmptyBase2 {};

struct SimplePadEmpty : public SimplePadEmptyBase1, SimplePadEmptyBase2 {
  // Bases have to occupy at least 1 byte of storage, so this requires
  // 2 bytes of padding, plus 1 byte for each base, yielding sizeof(T) = 8
  int32_t X;
} I;

struct SimplePadVfptr {
  virtual ~SimplePadVfptr() {}
  static void operator delete(void *ptr, size_t sz) {}
  int32_t X;
} J;

struct NonEmptyBase1 {
  bool X;
};

struct NonEmptyBase2 {
  bool Y;
};

struct SimplePadMultiInherit : public NonEmptyBase1, public NonEmptyBase2 {
  // X and Y from the 2 bases will get squished together, leaving 2 bytes
  // of padding necessary for proper alignment of an int32.
  // Therefore, sizeof(T) = 2 + 2 + 4 = 8
  int32_t X;
} K;

struct SimplePadMultiInherit2 : public SimplePadFields1, SimplePadFields2 {
  // There should be 1 byte of padding after the first class, and
  // 3 bytes of padding after the second class.
  int32_t X;
} L;

struct OneLevelInherit : public NonEmptyBase1 {
  short Y;
};

struct SimplePadTwoLevelInherit : public OneLevelInherit {
  // OneLevelInherit has nested padding because of its base,
  // and then padding again because of this class.  So each
  // class should be 4 bytes, yielding sizeof(T) = 12.
  int64_t Z;
} M;

struct SimplePadAggregate {
  NonEmptyBase1 X;
  int32_t Y;
  // the presence of X will cause 3 bytes of padding to be injected.
} N;

struct SimplePadVtable1 {
  static void operator delete(void *ptr, size_t sz) {}
  virtual ~SimplePadVtable1() {}
  virtual void A1() {}
  virtual void B1() {}
} O;

struct SimplePadVtable2 {
  static void operator delete(void *ptr, size_t sz) {}
  virtual ~SimplePadVtable2() {}
  virtual void X2() {}
  virtual void Y2() {}
  virtual void Z2() {}
} P;

struct SimplePadVtable3 {
  static void operator delete(void *ptr, size_t sz) {}
  virtual ~SimplePadVtable3() {}
  virtual void Foo3() {}
  virtual void Bar3() {}
  virtual void Baz3() {}
  virtual void Buzz3() {}
} Q;

struct SimplePadMultiVTables
    : public SimplePadVtable1,
      public SimplePadVtable2,
      public SimplePadVtable3 {

  ~SimplePadMultiVTables() override {}
  static void operator delete(void *ptr, size_t sz) {}

  // SimplePadVtable1 overrides
  void A1() override {}

  // SimplePadVtable2 overrides
  void Y2() override {}
  void Z2() override {}

  // SimplePadVtable3 overrides
  void Bar3() override {}
  void Baz3() override {}
  void Buzz3() override {}
} R;

int main(int argc, char **argv) {

  return 0;
}
