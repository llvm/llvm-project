// Compile with "cl /c /Zi /GR- /EHsc test-pdb-types.cpp"
// Link with "link test-pdb-types.obj /debug /nodefaultlib /entry:main
// /out:test-pdb-types.exe"

using namespace std;

// Sizes of builtin types
static const int sizeof_char = sizeof(char);
static const int sizeof_uchar = sizeof(unsigned char);
static const int sizeof_short = sizeof(short);
static const int sizeof_ushort = sizeof(unsigned short);
static const int sizeof_int = sizeof(int);
static const int sizeof_uint = sizeof(unsigned int);
static const int sizeof_long = sizeof(long);
static const int sizeof_ulong = sizeof(unsigned long);
static const int sizeof_longlong = sizeof(long long);
static const int sizeof_ulonglong = sizeof(unsigned long long);
static const int sizeof_int64 = sizeof(__int64);
static const int sizeof_uint64 = sizeof(unsigned __int64);
static const int sizeof_float = sizeof(float);
static const int sizeof_double = sizeof(double);
static const int sizeof_bool = sizeof(bool);
static const int sizeof_wchar = sizeof(wchar_t);

enum Enum {
  EValue1 = 1,
  EValue2 = 2,
};

enum ShortEnum : short { ESValue1 = 1, ESValue2 = 2 };

namespace NS {
class NSClass {
  float f;
  double d;
};
} // namespace NS

class Class {
public:
  class NestedClass {
    Enum e;
  };
  ShortEnum se;
};

int test_func(int a, int b) { return a + b; }

typedef Class ClassTypedef;
typedef NS::NSClass NSClassTypedef;
typedef int (*FuncPointerTypedef)();
typedef int (*VariadicFuncPointerTypedef)(char, ...);
FuncPointerTypedef GlobalFunc;
VariadicFuncPointerTypedef GlobalVariadicFunc;
int GlobalArray[10];

static const int sizeof_NSClass = sizeof(NS::NSClass);
static const int sizeof_Class = sizeof(Class);
static const int sizeof_NestedClass = sizeof(Class::NestedClass);
static const int sizeof_Enum = sizeof(Enum);
static const int sizeof_ShortEnum = sizeof(ShortEnum);
static const int sizeof_ClassTypedef = sizeof(ClassTypedef);
static const int sizeof_NSClassTypedef = sizeof(NSClassTypedef);
static const int sizeof_FuncPointerTypedef = sizeof(FuncPointerTypedef);
static const int sizeof_VariadicFuncPointerTypedef =
    sizeof(VariadicFuncPointerTypedef);
static const int sizeof_GlobalArray = sizeof(GlobalArray);

int main(int argc, char **argv) {
  ShortEnum e1;
  Enum e2;
  Class c1;
  Class::NestedClass c2;
  NS::NSClass c3;

  ClassTypedef t1;
  NSClassTypedef t2;
  return test_func(1, 2);
}
