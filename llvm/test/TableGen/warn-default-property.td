// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS -o /dev/null 2>&1 | FileCheck %s

include "llvm/IR/Intrinsics.td"

// CHECK: warning: property 'IntrWillReturn' is already enabled by default
// CHECK: warning: property 'IntrNoCallback' is already enabled by default
def int_foo : DefaultAttrsIntrinsic<[], [], [IntrWillReturn, IntrNoMem, IntrNoCallback]>;
