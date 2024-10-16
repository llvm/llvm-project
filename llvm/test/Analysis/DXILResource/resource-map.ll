; RUN: opt -S -disable-output -disable-output -passes="print<dxil-resource>" < %s 2>&1 | FileCheck %s

define void @test_typedbuffer() {
  ; RWBuffer<float4> Buf : register(u5, space3)
  %uav1 = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
              @llvm.dx.handle.fromBinding.tdx.TypedBuffer_f32_1_0(
                  i32 3, i32 5, i32 1, i32 0, i1 false)
  ; CHECK: Binding [[UAV1:[0-9]+]]:
  ; CHECK:   Symbol: ptr undef
  ; CHECK:   Name: ""
  ; CHECK:   Binding:
  ; CHECK:     Record ID: 0
  ; CHECK:     Space: 3
  ; CHECK:     Lower Bound: 5
  ; CHECK:     Size: 1
  ; CHECK:   Class: UAV
  ; CHECK:   Kind: TypedBuffer
  ; CHECK:   Globally Coherent: 0
  ; CHECK:   HasCounter: 0
  ; CHECK:   IsROV: 0
  ; CHECK:   Element Type: f32
  ; CHECK:   Element Count: 4

  ; CHECK:     Call bound to [[UAV1]]:  %uav1 = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0) @llvm.dx.handle.fromBinding.tdx.TypedBuffer_v4f32_1_0_0t(i32 3, i32 5, i32 1, i32 0, i1 false)
  ; CHECK-DAG: Resource [[UAV1]] is used by   %data0 = call <4 x float> @llvm.dx.typedBufferLoad.v4f32.tdx.TypedBuffer_v4f32_1_0_0t(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %uav1, i32 0)
  ; CHECK-DAG: Resource [[UAV1]] is used by   call void @llvm.dx.typedBufferStore.tdx.TypedBuffer_v4f32_1_0_0t.v4f32(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %uav1, i32 2, <4 x float> %data0)

  %data0 = call <4 x float> @llvm.dx.typedBufferLoad(
      target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %uav1, i32 0)
  call void @llvm.dx.typedBufferStore(
      target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %uav1,
      i32 2, <4 x float> %data0)

  ret void
}

