// Compile for x86 (FPO disabled)
// Compile with "cl /c /Zi /GR- PrettyFuncDumperTest.cpp"
// Link with "link PrettyFuncDumperTest.obj /debug /nodefaultlib /entry:main"

typedef void (*FuncPtrA)();
FuncPtrA FuncVarA;

typedef float (*FuncPtrB)(void);
FuncPtrB FuncVarB;

typedef int(*VariadicFuncPtrTypedef)(char, double, ...);
VariadicFuncPtrTypedef VariadicFuncVar;

void Func(int array[]) { return; }

template <int N=1, class ...T>
void TemplateFunc(T ...Arg) {
  return;
}

namespace {
  void Func(int& a, const double b, volatile bool c) { return; }
}

namespace NS {
  void Func(char a, int b, ...) {
    return;
  }
}

namespace MemberFuncsTest {
  class A {
  public:
    int FuncA() { return 1; }
    void FuncB(int a, ...) {}
  };
}

int main() {
  MemberFuncsTest::A v1;
  v1.FuncA();
  v1.FuncB(9, 10, 20);

  NS::Func('c', 2, 10, 100);

  TemplateFunc(10);
  TemplateFunc(10, 11, 88);
  return 0;
}
