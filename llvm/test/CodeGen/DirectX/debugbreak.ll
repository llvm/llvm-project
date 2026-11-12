; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.10-compute %s | FileCheck %s
; RUN: llc -O0 -mtriple=dxil-pc-shadermodel6.10-compute %s -o - | FileCheck %s
; RUN: llc -O2 -mtriple=dxil-pc-shadermodel6.10-compute %s -o - | FileCheck %s

; The test is using two instances of the debugBreak to make sure they will
; not get optimized away on -O2.

; CHECK-LABEL: main
; CHECK: call void @dx.op.debugBreak(i32 343)
; CHECK: call void @dx.op.debugBreak(i32 343)

define void @main() #0 {
entry:
  call void @llvm.debugtrap()
  call void @llvm.debugtrap()
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
