; RUN: not opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s
; CHECK: RWStructuredBuffers may increment or decrement their counters, but not both.

define void @inc_and_dec() {
entry:
  %handle = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding(i32 1, i32 2, i32 3, i32 4, ptr null)
  call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", float, 1, 0) %handle, i8 -1)
  call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", float, 1, 0) %handle, i8 1)
  ret void
}
