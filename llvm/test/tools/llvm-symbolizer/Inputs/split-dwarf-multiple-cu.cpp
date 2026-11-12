void f1();
__attribute__((always_inline)) inline void f2() {
  f1();
}
void f3() {
  f2();
}

// $ cat > other.cpp
// extern int i;
// int i;
// $ clang++ other.cpp split-dwarf-multiple-cu.cpp -g -c -Xclang \
//     -fdebug-compilation-dir -Xclang . -emit-llvm -S
// $ llvm-link other.ll split-dwarf-multiple-cu.ll -o split-dwarf-multiple-cu.bc
// $ clang++ -gsplit-dwarf split-dwarf-multiple-cu.bc -c
