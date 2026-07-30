 
// Global functions
int Func_arg_array(int array[]) { return 1; }
void Func_arg_void(void) { return; }
void Func_arg_none(void) { return; }
void Func_varargs(...) { return; }

// Class
namespace MemberTest {
  class A {
  public:
    int Func(int a, ...) { return 1; }
  };
}

// Template
template <int N=1, class ...T>
void TemplateFunc(T ...Arg) {
  return;
}

// namespace
namespace {
  void Func(int a, const long b, volatile bool c, ...) { return; }
}

namespace NS {
  void Func(char a, int b) {
    return;
  }
}

// Static function
static long StaticFunction(int a)
{
  return 2;
}

// Inlined function
inline void InlinedFunction(long a) { return; }

extern void FunctionCall();

int main() {
  MemberTest::A v1;
  v1.Func('a',10);

  Func(1, 5, true, 10, 8);
  NS::Func('c', 2);

  TemplateFunc(10);
  TemplateFunc(10,11,88);

  StaticFunction(2);
  InlinedFunction(1);

  FunctionCall();
  return 0;
}
