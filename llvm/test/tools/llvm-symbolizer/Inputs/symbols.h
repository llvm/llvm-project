// This file is a part of sources used to build `symbols.so`, which is used to
// test symbol location search made by llvm-symbolizer.
//
// Build instructions:
// $ mkdir /tmp/dbginfo
// $ cp symbols.h symbols.part1.cpp symbols.part2.cpp symbols.part3.c symbols.part4.c /tmp/dbginfo/
// $ cd /tmp/dbginfo
// $ gcc -osymbols.so -shared -fPIC -g symbols.part1.cpp symbols.part2.cpp symbols.part3.c symbols.part4.c


extern "C" {
extern int global_01;
int func_01();
int func_02(int);
}

template<typename T> T func_03(T x) {
  return x + T(1);
}
