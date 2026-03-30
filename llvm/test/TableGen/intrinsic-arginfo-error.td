// RUN: not llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s 2>&1 | FileCheck %s

include "llvm/IR/Intrinsics.td"

// CHECK: error: expected record type for the element with index 2 in list {{.*}}
def int_test : DefaultAttrsIntrinsic<
    [llvm_anyint_ty],
    [llvm_anyint_ty,
     LLVMMatchType<2>],
    [IntrNoMem]>;
