// Compile with "cl /c /Zi /GR- FilterTest.cpp"
// Link with "link FilterTest.obj /debug /nodefaultlib /entry:main"

class FilterTestClass {
public:
  typedef int NestedTypedef;
  enum NestedEnum {
    NestedEnumValue1
  };

  void MemberFunc() {}

private:
  int IntMemberVar;
  double DoubleMemberVar;
};

int IntGlobalVar;
double DoubleGlobalVar;
typedef int GlobalTypedef;
enum GlobalEnum {
  GlobalEnumVal1
} GlobalEnumVar;

int main(int argc, char **argv) {
  FilterTestClass TestClass;
  GlobalTypedef v1;
  return 0;
}
