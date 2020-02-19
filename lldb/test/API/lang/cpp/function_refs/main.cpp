// This is plagiarized from lit/SymbolFile/NativePDB/function-types-builtin.cpp.
void nullary() {}

template<typename Arg>
void unary(Arg) { }

template<typename A1, typename A2>
void binary(A1, A2) { }

int varargs(int, int, ...) { return 0; }

auto &ref = unary<bool>;
auto &ref2 = unary<volatile int*>;
auto &ref3 = varargs;
auto binp = &binary<int*, const int*>;
auto &binr = binary<int*, const int*>;
auto null = &nullary;
int main(int argc, char **argv) {
//% self.expect("target var ref", substrs=["(void (&)(bool))", "ref = 0x",
//%             "&::ref = <no summary available>"])
//% self.expect("target var ref2",
//%              substrs=["(void (&)(volatile int *))", "ref2 = 0x"])
//% self.expect("target var ref3",
//%              substrs=["(int (&)(int, int, ...))", "ref3 = 0x"])
//% self.expect("target var binp",
//%              substrs=["(void (*)(int *, const int *))", "binp = 0x"])
//% self.expect("target var binr",
//%              substrs=["(void (&)(int *, const int *))", "binr = 0x"])
//% self.expect("target var null",
//%              substrs=["(void (*)())", "null = 0x"])
  return 0;
}
