// This is a regression test that checks whether lldb can inspect the variables
// in this program without triggering an ASan exception.

__attribute__((noinline, optnone)) int use(int x) { return x; }

volatile int sink;

struct S1 {
  int f1;
  int *f2;
};

struct S2 {
  char a = 0;
  char b = 0;
  int pad = 0;
  S2(int x) {
    a = x & 0xff;
    b = x & 0xff00;
  }
};

int main() {
  S1 v1;
  v1.f1 = sink;
  v1.f2 = nullptr;
  sink++; //% self.expect("frame variable v1", substrs=["S1"])
  S2 v2(v1.f1);
  sink += use(v2.a); //% self.expect("frame variable v2", substrs=["S2"])
  sink += use(v2.pad); //% self.expect("frame variable v2", substrs=["S2"])
  return 0;
}
