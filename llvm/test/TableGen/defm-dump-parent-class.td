// RUN: llvm-tblgen %s -o - 2>&1 >/dev/null | FileCheck %s -DFILE=%s

// A dump in a multiclass body used to crash AddParentClass when the defm
// also had a class parent (defm X : MC, C;).

class C;

// dump next to a def.
multiclass MC1 {
// CHECK: [[FILE]]:[[@LINE+1]]:3: note: one
  dump "one";
  def D1;
}
defm X1 : MC1, C;

// dump as the only entry.
multiclass MC2 {
// CHECK: [[FILE]]:[[@LINE+1]]:3: note: two
  dump "two";
}
defm X2 : MC2, C;

// dump inside a foreach.
multiclass MC3 {
  foreach i = [1, 2] in {
// CHECK: [[FILE]]:[[@LINE+2]]:5: note: i = 1
// CHECK: [[FILE]]:[[@LINE+1]]:5: note: i = 2
    dump "i = " # i;
  }
  def D3;
}
defm X3 : MC3, C;

// anonymous defm.
multiclass MC4 {
// CHECK: [[FILE]]:[[@LINE+1]]:3: note: four
  dump "four";
  def D4;
}
defm : MC4, C;
