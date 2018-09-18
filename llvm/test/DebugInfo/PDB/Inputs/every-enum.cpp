// Build with "cl.exe /Zi /GR- /GX- every-enum.cpp /link /debug /nodefaultlib /incremental:no /entry:main"

#include <stdint.h>

// clang-format off
void *__purecall = 0;

void __cdecl operator delete(void *,unsigned int) {}
void __cdecl operator delete(void *,unsigned __int64) {}


enum I8 : int8_t {
  I8A = INT8_MIN,
  I8B = 0,
  I8C = INT8_MAX
};

enum I16 : int16_t {
  I16A = INT16_MIN,
  I16B = 0,
  I16C = INT16_MAX,
};

enum I32 : int32_t {
  I32A = INT32_MIN,
  I32B = 0,
  I32C = INT32_MAX,
};

enum I64 : int64_t {
  I64A = INT64_MIN,
  I64B = 0,
  I64C = INT64_MAX,
};

enum U8 : uint8_t {
  U8A = 0,
  U8B = UINT8_MAX
};

enum U16 : uint16_t {
  U16A = 0,
  U16B = UINT16_MAX,
};

enum U32 : uint32_t {
  U32A = 0,
  U32B = UINT32_MAX,
};

enum U64 : uint64_t {
  U64A = 0,
  U64B = UINT64_MAX,
};

enum Char16 : char16_t {
  C16A = u'a',
  C16B = u'b',
};

enum Char32 : char32_t {
  C32A = U'a',
  C32B = U'b',
};

enum WChar : wchar_t {
  WCA = L'a',
  WCB = L'b',
};

enum Bool : bool {
  BA = true,
  BB = false
};

enum class EC {
  A = 1,
  B = 2
};

struct Struct {
  enum Nested {
    A = 1,
    B = 2
  };
};

template<typename T> void f(T t) {}

int main(int argc, char **argv) {
  f(I8A);
  f(I16A);
  f(I32A);
  f(I64A);
  f(U8A);
  f(U16A);
  f(U32A);
  f(U64A);

  f(C16A);
  f(C32A);
  f(WCA);
  f(BA);


  f(EC::A);
  f(Struct::A);

  f<const volatile EC>(EC::A);
}
