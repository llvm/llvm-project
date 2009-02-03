// This is a regression test on debug info to make sure that we can get a
// meaningful stack trace from a C++ program.
// RUN: %llvmgcc -S -O0 -g %s -o - | llvm-as | llc --disable-fp-elim -o %t.s -f
// RUN: %compile_c %t.s -o %t.o
// RUN: %link %t.o -o %t.exe
// RUN: echo {break DeepStack::deepest\nrun 17\nwhere\n} > %t.in 
// RUN: gdb -q -batch -n -x %t.in %t.exe | tee %t.out | \
// RUN:   grep {#0  DeepStack::deepest.*(this=.*,.*x=33)}
// RUN: gdb -q -batch -n -x %t.in %t.exe | \
// RUN:   grep {#7  0x.* in main.*(argc=\[12\],.*argv=.*)}

// Only works on ppc, x86 and x86_64.  Should generalize?
// XFAIL: alpha|ia64|arm

#include <stdlib.h>

class DeepStack {
  int seedVal;
public:
  DeepStack(int seed) : seedVal(seed) {}

  int shallowest( int x ) { return shallower(x + 1); }
  int shallower ( int x ) { return shallow(x + 2); }
  int shallow   ( int x ) { return deep(x + 3); }
  int deep      ( int x ) { return deeper(x + 4); }
  int deeper    ( int x ) { return deepest(x + 6); }
  int deepest   ( int x ) { return x + 7; }

  int runit() { return shallowest(seedVal); }
};

int main ( int argc, char** argv) {

  DeepStack DS9( (argc > 1 ? atoi(argv[1]) : 0) );
  return DS9.runit();
}
