// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS 2>&1 | FileCheck %s -DFILE=%s

include "llvm/IR/Intrinsics.td"

// CHECK: [[FILE]]:[[@LINE+1]]:5: note: Explicitly specified name matches default name, consider dropping it
def int_foo0 : Intrinsic<[llvm_anyint_ty], [], [], "llvm.foo0">;
