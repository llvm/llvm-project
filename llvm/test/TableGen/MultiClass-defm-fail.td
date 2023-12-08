// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s
// XFAIL: vg_leak

// This test verifies that tablegen does fail if it can't resolve an unresolved
// !cast() during processing top-level defm.

class A {}
class B<A a> {
  A ba = a;
}

multiclass M0<string s> {
  // This should work fine.
  def _m00 : B<!cast<A>(s)>;
  // CHECK: error: Undefined reference to record: 'd1_r1_no_such_record'
  def _m01: B<!cast<A>(s#"_no_such_record")>;
}

multiclass M1<string s> {
  def _r1 : A;
  // It would be nice if we could refer to _r1's name without having to pass it
  // explicitly via 's'.
  // XCHECK-DAG: note: instantiated from multiclass
  defm _m1: M0<s # "_r1">;
}

// CHECK: def _m01: B
// CHECK: note: instantiated from multiclass
// CHECK: defm _m1: M0
// CHECK: note: instantiated from multiclass
// CHECK: defm d1: M1
defm d1: M1<"d1">;
