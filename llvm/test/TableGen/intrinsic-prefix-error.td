// RUN: not llvm-tblgen -gen-intrinsic-enums --intrinsic-prefix=gen3 -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS 2>&1 | FileCheck %s

include "llvm/IR/Intrinsics.td"

// CHECK: error: tried to generate intrinsics for unknown target gen3
// CHECK-NEXT: Known targets are: gen1, gen2

let TargetPrefix = "gen1" in {
  def int_gen1_int0 : Intrinsic<[llvm_i32_ty]>;
}

let TargetPrefix = "gen2" in {
  def int_gen2_int0 : Intrinsic<[llvm_i32_ty]>;
}
