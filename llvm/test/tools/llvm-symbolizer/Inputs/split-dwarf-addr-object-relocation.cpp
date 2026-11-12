void f1();
__attribute__((always_inline)) void f2() {
  f1();
}
void f3() {
  f2();
}

// $ clang++ split-dwarf-addr-object-relocation.cpp -gsplit-dwarf -c Xclang \
//     -fdebug-compilation-dir -Xclang .
