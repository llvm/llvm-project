// Build with "cl.exe /Z7 /GR- /GS- /GX- every-class.cpp /link /debug:full /nodefaultlib /incremental:no /entry:main"

#include <stdint.h>

// clang-format off
void *__purecall = 0;

void __cdecl operator delete(void *, unsigned int) {}
void __cdecl operator delete(void *, unsigned __int64) {}

struct Nothing {};
struct Constructor { Constructor() {} };
struct Assignment {
  Assignment &operator=(Assignment Other) { return *this; }
};
struct Cast {
  operator int() { return 42; }
};

struct Nested {
  struct F {};
};
struct Operator {
  int operator+(int X) { return 42; }
};

class Class {};

union Union {};

enum class Enum {A};


template<typename T> void f(T t) {}

int main(int argc, char **argv) {
  struct Scoped {};
  
  struct { } Anonymous;

  f(Nothing{});
  f(Constructor{});
  f(Assignment{});
  f(Cast{});
  f(Nested{});
  f(Operator{});
  f(Nested::F{});
  f(Scoped{});
  f(Class{});
  f(Union{});
  f(Anonymous);
  f(Enum::A);
  

  f<const Nothing>(Nothing{});
  f<volatile Nothing>(Nothing{});
  f<const volatile Nothing>(Nothing{});
  f<__unaligned Nothing>(Nothing{});

  return 0;
}
