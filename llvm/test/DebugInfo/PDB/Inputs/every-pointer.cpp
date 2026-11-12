// Build with "cl.exe /Zi /GR- /GX- every-pointer.cpp /link /debug /nodefaultlib /incremental:no /entry:main"

#include <stdint.h>

// clang-format off
void *__purecall = 0;

void __cdecl operator delete(void *,unsigned int) {}
void __cdecl operator delete(void *,unsigned __int64) {}


struct Foo {
  int X = 0;
  int func() { return 42; }
};

int *IntP = nullptr;
Foo *FooP = nullptr;

Foo F;

Foo __unaligned *UFooP = &F;
Foo * __restrict RFooP = &F;

const Foo * CFooP = &F;
volatile Foo * VFooP = &F;
const volatile Foo * CVFooP = &F;

template<typename T> void f(T t) {}

int main(int argc, char **argv) {
  f<int*>(IntP);
  f<Foo*>(FooP);
  
  f<Foo __unaligned *>(UFooP);
  f<Foo *__restrict>(RFooP);
  
  f<const Foo*>(CFooP);
  f<volatile Foo*>(VFooP);
  f<const volatile Foo*>(CVFooP);
  
  f<Foo&>(F);
  f<Foo&&>(static_cast<Foo&&>(F));
  
  f(&Foo::X);
  f(&Foo::func);
  return 0;
}
