// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST0 2>&1 | FileCheck %s --check-prefix=CHECK-TEST0
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST1 2>&1 | FileCheck %s --check-prefix=CHECK-TEST1
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST2 2>&1 | FileCheck %s --check-prefix=CHECK-TEST2
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST3 2>&1 | FileCheck %s --check-prefix=CHECK-TEST3
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST4 2>&1 | FileCheck %s --check-prefix=CHECK-TEST4

// This unit test tests whether invalid use of `llvm_vararg_ty` are flagged.

#define TEST_INTRINSICS_SUPPRESS_DEFS
include "llvm/IR/Intrinsics.td"

#ifdef TEST0
// CHECK-TEST0: error: cannot use llvm_vararg_ty as a return type
def int_test: Intrinsic<[llvm_vararg_ty], []>;
#endif // TEST0

#ifdef TEST1
// CHECK-TEST1: error: cannot use llvm_vararg_ty as a return type
def int_test: Intrinsic<[llvm_i32_ty, llvm_vararg_ty], []>;
#endif // TEST1

#ifdef TEST2
// CHECK-TEST2: error: llvm_vararg_ty can only be the last parameter type
def int_test: Intrinsic<[], [llvm_i32_ty, llvm_vararg_ty, llvm_i1_ty]>;
#endif // TEST2

#ifdef TEST3
// CHECK-TEST3: error: llvm_vararg_ty can only be the last parameter type
def int_test: Intrinsic<[], [llvm_i32_ty, llvm_vararg_ty, llvm_vararg_ty]>;
#endif // TEST3

#ifdef TEST4
// CHECK-TEST4: error: llvm_vararg_ty can only be the last parameter type
def int_test: Intrinsic<[], [llvm_vararg_ty, llvm_i32_ty, llvm_vararg_ty]>;
#endif // TEST4
