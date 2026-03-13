; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: %5:vfid(<2 x s64>) = nnan ninf nsz arcp afn reassoc G_INTRINSIC intrinsic(@llvm.spv.unpackhalf2x16), %0:iid(s64) is only supported with the GLSL extended instruction set.

define hidden spir_func noundef nofpclass(nan inf) float @_Z9test_funcj(i32 noundef %0) local_unnamed_addr #0 {
  %2 = tail call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.spv.unpackhalf2x16.v2f32(i32 %0)
  %3 = extractelement <2 x float> %2, i64 0
  ret float %3
}

