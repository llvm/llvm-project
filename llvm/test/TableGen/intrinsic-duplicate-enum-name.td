// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS 2>&1 | FileCheck %s -DFILE=%s

include "llvm/IR/Intrinsics.td"

def int_x__y_z : Intrinsic<[llvm_anyint_ty], [], []>;

// CHECK: [[FILE]]:[[@LINE+2]]:5: error: `Intrinsic::x_y_z` is already defined
// CHECK: [[FILE]]:[[@LINE-3]]:5: note: Previous definition here
def int_x_y_z : Intrinsic<[llvm_anyint_ty], [], []>;
