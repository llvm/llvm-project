; RUN: opt -S -disable-output -disable-output -passes="print<dxil-resource>,dxil-op-lower,print<dxil-resource>" -mtriple=dxil-pc-shadermodel6.6-compute < %s 2>&1 | FileCheck %s -check-prefixes=CHECK,CHECK_SM66
; RUN: opt -S -disable-output -disable-output -passes="print<dxil-resource>,dxil-op-lower,print<dxil-resource>" -mtriple=dxil-pc-shadermodel6.2-compute < %s 2>&1 | FileCheck %s -check-prefixes=CHECK,CHECK_SM62

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

  ;
  ;;; After dxil-op-lower, the DXILResourceMap info should be updated.
  ;
  ; CHECK_SM66:     Call bound to [[UAV1]]:  %uav11 = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 218, %dx.types.ResBind { i32 5, i32 5, i32 3, i8 1 }, i32 0, i1 false)
  ; CHECK_SM66-DAG: Resource [[UAV1]] is used by   %data02 = call %dx.types.ResRet.f32 @dx.op.bufferLoad.f32(i32 68, %dx.types.Handle %uav1_annot, i32 0, i32 undef)
  ; CHECK_SM66-DAG: Resource [[UAV1]] is used by   call void @dx.op.bufferStore.f32(i32 69, %dx.types.Handle %uav1_annot, i32 2, i32 undef, float %9, float %10, float %11, float %12, i8 15)
  ;
  ; CHECK_SM62:     Call bound to [[UAV1]]:  %uav11 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 0, i1 false)
  ; CHECK_SM62-DAG: Resource [[UAV1]] is used by   %data02 = call %dx.types.ResRet.f32 @dx.op.bufferLoad.f32(i32 68, %dx.types.Handle %uav11, i32 0, i32 undef)
  ; CHECK_SM62-DAG: Resource [[UAV1]] is used by   call void @dx.op.bufferStore.f32(i32 69, %dx.types.Handle %uav11, i32 2, i32 undef, float %9, float %10, float %11, float %12, i8 15)

  ret void
}

