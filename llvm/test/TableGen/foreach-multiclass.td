// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: --- Defs ---

// CHECK: def A00 {
// CHECK:   int sum = 7;
// CHECK: }

// CHECK: def A01 {
// CHECK:   int sum = 8;
// CHECK: }

// CHECK-NOT: def B0

// CHECK: def B12 {
// CHECK:   int val = 9;
// CHECK: }

// CHECK: def B20 {
// CHECK:   int val = 7;
// CHECK: }

// CHECK: def B24 {
// CHECK:   int val = 11;
// CHECK: }

// CHECK: def B25 {
// CHECK:   int val = 12;
// CHECK: }

// CHECK: def C04
// CHECK: def C05

// CHECK: def D0A
// CHECK-NOT: def D0B
// CHECK: def D1A
// CHECK: def D1B

// CHECK: def E01
// CHECK: def E02
// CHECK-NOT: def E0C

// CHECK: def E18
// CHECK: def E19
// CHECK: def E1C33
// CHECK: def E1C34
// CHECK: def E1C55
// CHECK: def E1C56

// CHECK-NOT: def F0
// CHECK-NOT: def F1
// CHECK-NOT: def F2_0_0
// CHECK:     def F2_1_0
// CHECK-NOT: def F2_1_2
// CHECK:     def F2_2_0
// CHECK:     def F2_2_1
// CHECK-NOT: def F2_2_2

// CHECK: def G0
// CHECK: def H0_G0_0

multiclass A<int x> {
  foreach i = [0, 1] in {
    def NAME#i {
      int sum = !add(x, i);
    }
  }
}

defm A0 : A<7>;

multiclass B<int x, list<int> lst> {
  foreach i = lst in {
    def NAME#i {
      int val = !add(x, i);
    }
  }
}

defm B0 : B<7, []>;
defm B1 : B<7, [2]>;
defm B2 : B<7, [0, 4, 5]>;

multiclass C<int x> {
  foreach i = [x, !add(x, 1)] in {
    def NAME#i;
  }
}

defm C0 : C<4>;

multiclass D<bit b> {
  def A;

  foreach _ = !if(b, [0], []<int>) in
  def B;
}

defm D0 : D<0>;
defm D1 : D<1>;

multiclass E<list<int> lst, int x>
  : C<x> {
  foreach i = lst in
  defm C#i : C<i>;
}

defm E0 : E<[], 1>;
defm E1 : E<[3, 5], 8>;

multiclass F<list<int> lst> {
  foreach i = lst in
  foreach j = !foldl([]<int>, lst, lhs, x,
                     !if(!lt(x, i), !listconcat(lhs, [x]), lhs)) in
  def _#i#_#j;
}

defm F0 : F<[]>;
defm F1 : F<[0]>;
defm F2 : F<[0, 1, 2]>;

// If multiclass argument comes from loop variable,
// and field of argument is placed at foreach statement,
// the record field must be resolved correctly.
class G {
  list<int> val = [0];
}

multiclass H<G g> {
  foreach n = g.val in
    def _#g#_#n;
}

def G0 : G;

foreach g = [G0] in
  defm H0 : H<g>;
