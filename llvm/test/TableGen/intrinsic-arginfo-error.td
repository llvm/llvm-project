// RUN: not llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s 2>&1 | FileCheck %s

include "llvm/IR/Intrinsics.td"

// CHECK: error: assertion failed: dependent types must reference an overload index of an 'llvm_any' type
def int_test : DefaultAttrsIntrinsic<
    [llvm_anyint_ty],
    [llvm_anyint_ty, LLVMMatchType<2>],
    [IntrNoMem]>;
