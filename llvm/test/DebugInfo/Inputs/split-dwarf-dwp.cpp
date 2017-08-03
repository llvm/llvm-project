void f1();
__attribute__((always_inline)) void f2() {
  f1();
}
void f3() {
  f2();
}

To produce split-dwarf-dwp.o{,dwp}, run:

  $ clang++ split-dwarf-dwp.cpp -gsplit-dwarf -c -Xclang -fdebug-compilation-dir=Output -fno-split-dwarf-inlining
  $ llvm-dwp split-dwarf-dwp.dwo -o split-dwarf-dwp.o.dwp
