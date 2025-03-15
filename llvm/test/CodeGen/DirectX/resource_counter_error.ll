; RUN: not opt -S -passes='dxil-op-lower' -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s
; CHECK: <unknown>:0:0: RWStructuredBuffers may increment or decrement their counters, but not both.

define void @inc_and_dec() {
entry:
  %handle = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 1, i32 2, i32 3, i32 4, i1 false)
  call i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %handle, i8 -1)
  call i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %handle, i8 1)
  ret void
}

declare target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32, i32, i32, i32, i1)
declare i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0), i8)
