// Build with "cl.exe /Zi /GR- /GX- every-array.cpp /link /debug /nodefaultlib /entry:main"

// clang-format off
void *__purecall = 0;

void __cdecl operator delete(void *,unsigned int) {}
void __cdecl operator delete(void *,unsigned __int64) {}


int func1() { return 42; }
int func2() { return 43; }
int func3() { return 44; }

template<typename T>
void Reference(T &t) { }

int IA[3] = {1, 2, 3};
const int CIA[3] = {1, 2, 3};
volatile int VIA[3] = {1, 2, 3};

using FuncPtr = decltype(&func1);
FuncPtr FA[3] = {&func1, &func2, &func3};

struct S {
  int N;
  int f() const { return 42; }
};

using MemDataPtr = decltype(&S::N);
using MemFunPtr = decltype(&S::f);

MemDataPtr MDA[1] = {&S::N};
MemFunPtr MFA[1] = {&S::f};


int main(int argc, char **argv) {
}
