void f1() {}
__attribute__((always_inline)) inline void f2() {
  f1();
}
int main() {
  f2();
}

// $ clang++ split-dwarf-test.cpp -gsplit-dwarf -Xclang \
//     -fdebug-compilation-dir -Xclang . -o split-dwarf-test
// $ clang++ split-dwarf-test.cpp -gsplit-dwarf -Xclang \
//     -fdebug-compilation-dir -Xclang . -fno-split-dwarf-inlining \
//     -o split-dwarf-test-nogmlt
