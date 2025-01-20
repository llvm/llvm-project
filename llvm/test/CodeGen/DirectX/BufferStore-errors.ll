; We use llc for this test so that we don't abort after the first error.
; RUN: not llc %s -o /dev/null 2>&1 | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK: error:
; CHECK-SAME: in function storetoomany
; CHECK-SAME: typedBufferStore data must be a vector of 4 elements
define void @storetoomany(<5 x float> %data, i32 %index) "hlsl.export" {
  %buffer = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v4f32_1_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  call void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_v4f32_1_0_0t.v5f32(
      target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %buffer,
      i32 %index, <5 x float> %data)

  ret void
}

; CHECK: error:
; CHECK-SAME: in function storetoofew
; CHECK-SAME: typedBufferStore data must be a vector of 4 elements
define void @storetoofew(<3 x i32> %data, i32 %index) "hlsl.export" {
  %buffer = call target("dx.TypedBuffer", <4 x i32>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v4i32_1_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  call void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_v4i32_1_0_0t.v3i32(
      target("dx.TypedBuffer", <4 x i32>, 1, 0, 0) %buffer,
      i32 %index, <3 x i32> %data)

  ret void
}

declare void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_v4f32_1_0_0t.v5f32(target("dx.TypedBuffer", <4 x float>, 1, 0, 0), i32, <5 x float>)
declare void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_v4i32_1_0_0t.v3i32(target("dx.TypedBuffer", <4 x i32>, 1, 0, 0), i32, <3 x i32>)
