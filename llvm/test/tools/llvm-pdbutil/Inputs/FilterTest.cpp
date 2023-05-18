// Compile with "cl /c /Zi /GR- FilterTest.cpp"
// Link with "link FilterTest.obj /debug /nodefaultlib /entry:main"

class FilterTestClass {
public:
  typedef int NestedTypedef;
  enum NestedEnum {
    NestedEnumValue1
  };

  void MemberFunc() {}

  int foo() const { return IntMemberVar; }

private:
  int IntMemberVar;
  double DoubleMemberVar;
};

int IntGlobalVar;
double DoubleGlobalVar;
typedef int GlobalTypedef;
char OneByte;
char TwoBytes[2];
char ThreeBytes[3];

enum GlobalEnum {
  GlobalEnumVal1
} GlobalEnumVar;

int CFunc() {
  return (int)OneByte * 2;
}
int BFunc() {
  return 42;
}
int AFunc() {
  static FilterTestClass FC;

  return (CFunc() + BFunc()) * IntGlobalVar + FC.foo();
}

int main(int argc, char **argv) {
  FilterTestClass TestClass;
  GlobalTypedef v1;
  return 0;
}
