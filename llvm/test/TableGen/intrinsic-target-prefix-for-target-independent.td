// RUN: not llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s 2>&1 | FileCheck %s -DFILE=%s

include "llvm/IR/Intrinsics.td"

// Check that target independent intrinsics with a prefix that matches a target
// name are flagged.
// CHECK: [[FILE]]:[[@LINE+1]]:5: error: target independent intrinsic `llvm.aarch64.foo' has prefix `llvm.aarch64` that conflicts with intrinsics for target `aarch64`
def int_aarch64_foo : Intrinsic<[],[]>;

