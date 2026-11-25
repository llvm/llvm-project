; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown < %s -o - 2>&1 | FileCheck %s
; RUN: not llc -O2 -mtriple=spirv64-unknown-unknown < %s -o - 2>&1 | FileCheck %s
; CHECK: error: {{.*}}intrinsic selection not implemented
; CHECK-NOT: Assertion

define spir_func <2 x i32> @sdiv_v2i32(<2 x i32> %a, <2 x i32> %b, <2 x i1> %m) {
  %r = call <2 x i32> @llvm.masked.sdiv.v2i32(<2 x i32> %a, <2 x i32> %b, <2 x i1> %m)
  ret <2 x i32> %r
}
