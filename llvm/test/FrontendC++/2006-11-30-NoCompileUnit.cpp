// This is a regression test on debug info to make sure we don't hit a compile 
// unit size issue with gdb.
// RUN: %llvmgcc -S -O0 -g %s -o - | llvm-as | \
// RUN:   llc --disable-fp-elim -o NoCompileUnit.s -f
// RUN: %compile_c NoCompileUnit.s -o NoCompileUnit.o
// RUN: %link NoCompileUnit.o -o NoCompileUnit.exe
// RUN: echo {break main\nrun\np NoCompileUnit::pubname} > %t2
// RUN: gdb -q -batch -n -x %t2 NoCompileUnit.exe | \
// RUN:   tee NoCompileUnit.out | not grep {"low == high"}
// XFAIL: alpha|ia64|arm
// XFAIL: *
// See PR2454


class MamaDebugTest {
private:
  int N;
  
protected:
  MamaDebugTest(int n) : N(n) {}
  
  int getN() const { return N; }

};

class BabyDebugTest : public MamaDebugTest {
private:

public:
  BabyDebugTest(int n) : MamaDebugTest(n) {}
  
  static int doh;
  
  int  doit() {
    int N = getN();
    int Table[N];
    
    int sum = 0;
    
    for (int i = 0; i < N; ++i) {
      int j = i;
      Table[i] = j;
    }
    for (int i = 0; i < N; ++i) {
      int j = Table[i];
      sum += j;
    }
    
    return sum;
  }

};

int BabyDebugTest::doh;


int main(int argc, const char *argv[]) {
  BabyDebugTest BDT(20);
  return BDT.doit();
}
