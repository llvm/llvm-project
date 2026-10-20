// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST0 2>&1 | FileCheck %s --check-prefix=CHECK-TEST0
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST1 2>&1 | FileCheck %s --check-prefix=CHECK-TEST1
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST2 2>&1 | FileCheck %s --check-prefix=CHECK-TEST2
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST3 2>&1 | FileCheck %s --check-prefix=CHECK-TEST3

#define TEST_INTRINSICS_SUPPRESS_DEFS
include "llvm/IR/Intrinsics.td"

// Check for invalid overload index.

#ifdef TEST0
// Overload index inferred for argument type.

// CHECK-TEST0: error: assertion failed: cannot support more than 256 overload types
def int_tes : Intrinsic<[], !listsplat(llvm_anyint_ty, 257)>;
#endif // TEST0

#ifdef TEST1
// Overload index inferred for return type.

// CHECK-TEST1: error: assertion failed: cannot support more than 256 overload types
def int_test : Intrinsic<!listsplat(llvm_anyint_ty, 257),[]>;
#endif // TEST1

#ifdef TEST2
// Overload index OOR with combination of argument and return types.

// CHECK-TEST2: error: assertion failed: cannot support more than 256 overload types
def int_test : Intrinsic<!listsplat(llvm_anyint_ty, 250),
                         !listsplat(llvm_anyint_ty, 7)>;
#endif // TEST2

#ifdef TEST3
// Invalid negative overload index.
// CHECK-TEST3: rror: assertion failed: overload index must be >= 0
def int_test : Intrinsic<[llvm_anyint_ty],
                         [LLVMMatchType<-1>]>;
#endif // TEST3
