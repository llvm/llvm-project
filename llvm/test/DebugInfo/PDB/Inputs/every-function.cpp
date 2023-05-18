// Build with "cl.exe /Zi /GR- /GS- -EHs-c- every-function.cpp /link /debug /nodefaultlib /incremental:no /entry:main"
// Getting functions with the correct calling conventions requires building in x86.

// clang-format off
void *__purecall = 0;

void __cdecl operator delete(void *,unsigned int) {}
void __cdecl operator delete(void *,unsigned __int64) {}

// All calling conventions that appear in normal code.
int __cdecl cc_cdecl() { return 42; }
int __stdcall cc_stdcall() { return 42; }
int __fastcall cc_fastcall() { return 42; }
int __vectorcall cc_vectorcall() { return 42; }


struct Struct {
  Struct() {}  // constructor

  int __thiscall cc_thiscall() { return 42; }

  void M() { }
  void CM() const { }
  void VM() volatile { }
  void CVM() const volatile { }
};

int builtin_one_param(int x) { return 42; }
int builtin_two_params(int x, char y) { return 42; }

void struct_one_param(Struct S) { }

void modified_builtin_param(const int X) { }
void modified_struct_param(const Struct S) { }

void pointer_builtin_param(int *X) { }
void pointer_struct_param(Struct *S) { }


void modified_pointer_builtin_param(const int *X) { }
void modified_pointer_struct_param(const Struct *S) { }

Struct rvo() { return Struct(); }

struct Base1 {
  virtual ~Base1() {}
};

struct Base2 : public virtual Base1 { };

struct Derived : public virtual Base1, public Base2 {
};


int main() {
  cc_cdecl();
  cc_stdcall();
  cc_fastcall();
  Struct().cc_thiscall();
  cc_vectorcall();

  builtin_one_param(42);
  builtin_two_params(42, 'x');
  struct_one_param(Struct{});

  modified_builtin_param(42);
  modified_struct_param(Struct());

  pointer_builtin_param(nullptr);
  pointer_struct_param(nullptr);


  modified_pointer_builtin_param(nullptr);
  modified_pointer_struct_param(nullptr);

  Struct S = rvo();

  Derived D;
  return 42;
}
