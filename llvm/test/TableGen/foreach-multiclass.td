// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: --- Defs ---

// CHECK: def A00 {
// CHECK:   int sum = 7;
// CHECK: }

// CHECK: def A01 {
// CHECK:   int sum = 8;
// CHECK: }

multiclass A<int x> {
  // Allow foreach in multiclass as long as the list does not depend on
  // template args.
  foreach i = [0, 1] in {
    def NAME#i {
      int sum = !add(x, i);
    }
  }
}

defm A0 : A<7>;
