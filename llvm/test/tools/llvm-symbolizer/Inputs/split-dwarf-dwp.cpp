void f1() {}
__attribute__((always_inline)) inline void f2() {
  f1();
}
// throw a gap in the address range to force use of DW_AT_ranges, ranges_base,
// range contribution in the .dwo file, etc.
__attribute__((nodebug)) void gap() {
}
int main() {
  f2();
}

// To produce split-dwarf-dwp.o{,dwp}, Create another file that has ranges, so
// the ranges_base of the CU for split-dwarf-dwp.cpp is non-zero.
//
//   $ cat > other.cpp
//   void other1() {}
//   __attribute__((nodebug)) void other2() {}
//   void other3() {}
//   $ clang++ other.cpp split-dwarf-dwp.cpp -gsplit-dwarf -c -Xclang -fdebug-compilation-dir -Xclang . -fno-split-dwarf-inlining
//   $ llvm-dwp other.dwo split-dwarf-dwp.dwo -o test/DebugInfo/Inputs/split-dwarf-dwp.o.dwp
//   $ ld -r other.o split-dwarf-dwp.o -o test/DebugInfo/Inputs/split-dwarf-dwp.o
