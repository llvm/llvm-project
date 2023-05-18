// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

class A {}
class B<A a> {
  A ba = a;
}

multiclass M0<string s> {
  def _m0 : B<!cast<A>(s)>;

  // Uncomment to test that !cast will eventually fail if the record it refers
  // to does not exist by the time we instantiate this record from the top
  // level.
  //def _m1 : B<!cast<A>(s#"X")>;
}

multiclass M1<string s> {
  def _r1 : A;
  // It would be nice if we could refer to _r1's name without having to pass it
  // explicitly via 's'.
  defm _m1: M0<s # "_r1">;
}

multiclass M2 {
  def _x : A {
    string n = NAME;
  }

  def _y : B<!cast<A>(NAME # "_x")>;

  // This used to throw an error during multiclass parsing as NAME was not
  // recognized when parsing the template arguments.
  defm NAME: M1<NAME>;
}
defm d0: M2;
// CHECK-LABEL: def d0_m1_m0
// CHECK: A ba = d0_r1;
//
// CHECK-LABEL: def d0_x {
// CHECK: string n = "d0";
//
// CHECK-LABEL: def d0_y {
// CHECK: A ba = d0_x;

// This always works, because d1_r1 is instantiated before d1_m1 which would
// attempt to !cast to it.
defm d1: M1<"d1">;
// CHECK-LABEL: def d1_m1_m0
// CHECK: A ba = d1_r1;
