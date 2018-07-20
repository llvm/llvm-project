// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: --- Defs ---

// CHECK: def Sum {
// CHECK:   int x = 712;
// CHECK: }

// CHECK: def yyy_A0
// CHECK: def yyy_A1
// CHECK: def yyy_A2
// CHECK: def yyy_B0A0
// CHECK: def yyy_B0A1
// CHECK: def yyy_C0B0A0
// CHECK: def yyy_C0B0A1
// CHECK: def yyy_C0B1A0
// CHECK: def yyy_C0B1A1
// CHECK-NOT: def zzz_A0
// CHECK: def zzz_B0A0
// CHECK: def zzz_B0A1
// CHECK: def zzz_C0B0A0
// CHECK: def zzz_C0B0A1
// CHECK: def zzz_C0B1A0
// CHECK: def zzz_C0B1A1

class A<int a> {
  int Num = a;
}

multiclass B<int b> {
  def A0 : A<!add(10, b)>;
  def A1 : A<!add(20, b)>;
}

multiclass C<int c> {
  defm B0 : B<!add(100, c)>;
  defm B1 : B<!add(200, c)>;
}

defset list<A> As = {
  def A0 : A<1>;
  foreach i = 1-2 in {
    def A#i : A<!add(i, 1)>;
  }
  defset list<A> SubAs = {
    defm B0 : B<2>;
    defm C0 : C<3>;
  }
}

def Sum {
  int x = !foldl(0, As, a, b, !add(a, b.Num));
}

foreach a = As in {
  def yyy_ # !cast<string>(a);
}

foreach a = SubAs in {
  def zzz_ # !cast<string>(a);
}
