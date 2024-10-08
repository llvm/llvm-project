; RUN: opt -S -disable-output -passes="print<dxil-resource>" < %s 2>&1 | FileCheck %s

@G = external constant <4 x float>, align 4

define void @test_typedbuffer() {
  ; ByteAddressBuffer Buf : register(t8, space1)
  %srv0 = call target("dx.RawBuffer", i8, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.RawBuffer_i8_0_0t(
          i32 1, i32 8, i32 1, i32 0, i1 false)
  ; CHECK: Binding [[SRV0:[0-9]+]]:
  ; CHECK:   Symbol: ptr undef
  ; CHECK:   Name: ""
  ; CHECK:   Binding:
  ; CHECK:     Record ID: 0
  ; CHECK:     Space: 1
  ; CHECK:     Lower Bound: 8
  ; CHECK:     Size: 1
  ; CHECK:   Class: SRV
  ; CHECK:   Kind: RawBuffer

  ; struct S { float4 a; uint4 b; };
  ; StructuredBuffer<S> Buf : register(t2, space4)
  %srv1 = call target("dx.RawBuffer", {<4 x float>, <4 x i32>}, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.RawBuffer_sl_v4f32v4i32s_0_0t(
          i32 4, i32 2, i32 1, i32 0, i1 false)
  ; CHECK: Binding [[SRV1:[0-9]+]]:
  ; CHECK:   Symbol: ptr undef
  ; CHECK:   Name: ""
  ; CHECK:   Binding:
  ; CHECK:     Record ID: 1
  ; CHECK:     Space: 4
  ; CHECK:     Lower Bound: 2
  ; CHECK:     Size: 1
  ; CHECK:   Class: SRV
  ; CHECK:   Kind: StructuredBuffer
  ; CHECK:   Buffer Stride: 32
  ; CHECK:   Alignment: 4

  ; Buffer<uint4> Buf[24] : register(t3, space5)
  %srv2 = call target("dx.TypedBuffer", <4 x i32>, 0, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.TypedBuffer_i32_0_0t(
          i32 5, i32 3, i32 24, i32 0, i1 false)
  ; CHECK: Binding [[SRV2:[0-9]+]]:
  ; CHECK:   Symbol: ptr undef
  ; CHECK:   Name: ""
  ; CHECK:   Binding:
  ; CHECK:     Record ID: 2
  ; CHECK:     Space: 5
  ; CHECK:     Lower Bound: 3
  ; CHECK:     Size: 24
  ; CHECK:   Class: SRV
  ; CHECK:   Kind: TypedBuffer
  ; CHECK:   Element Type: u32
  ; CHECK:   Element Count: 4

  ; RWBuffer<int> Buf : register(u7, space2)
  %uav0 = call target("dx.TypedBuffer", i32, 1, 0, 1)
      @llvm.dx.handle.fromBinding.tdx.TypedBuffer_i32_1_0t(
          i32 2, i32 7, i32 1, i32 0, i1 false)
  ; CHECK: Binding [[UAV0:[0-9]+]]:
  ; CHECK:   Symbol: ptr undef
  ; CHECK:   Name: ""
  ; CHECK:   Binding:
  ; CHECK:     Record ID: 0
  ; CHECK:     Space: 2
  ; CHECK:     Lower Bound: 7
  ; CHECK:     Size: 1
  ; CHECK:   Class: UAV
  ; CHECK:   Kind: TypedBuffer
  ; CHECK:   Globally Coherent: 0
  ; CHECK:   HasCounter: 0
  ; CHECK:   IsROV: 0
  ; CHECK:   Element Type: i32
  ; CHECK:   Element Count: 1

  ; RWBuffer<float4> Buf : register(u5, space3)
  %uav1 = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
              @llvm.dx.handle.fromBinding.tdx.TypedBuffer_f32_1_0(
                  i32 3, i32 5, i32 1, i32 0, i1 false)
  ; CHECK: Binding [[UAV1:[0-9]+]]:
  ; CHECK:   Symbol: ptr undef
  ; CHECK:   Name: ""
  ; CHECK:   Binding:
  ; CHECK:     Record ID: 1
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

  ; RWBuffer<float4> BufferArray[10] : register(u0, space4)
  ; RWBuffer<float4> Buf = BufferArray[0]
  %uav2_1 = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
              @llvm.dx.handle.fromBinding.tdx.TypedBuffer_f32_1_0(
                  i32 4, i32 0, i32 10, i32 0, i1 false)
  ; RWBuffer<float4> Buf = BufferArray[5]
  %uav2_2 = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
              @llvm.dx.handle.fromBinding.tdx.TypedBuffer_f32_1_0(
                  i32 4, i32 0, i32 10, i32 5, i1 false)
  ; CHECK: Binding [[UAV2:[0-9]+]]:
  ; CHECK:   Symbol: ptr undef
  ; CHECK:   Name: ""
  ; CHECK:   Binding:
  ; CHECK:     Record ID: 2
  ; CHECK:     Space: 4
  ; CHECK:     Lower Bound: 0
  ; CHECK:     Size: 10
  ; CHECK:   Class: UAV
  ; CHECK:   Kind: TypedBuffer
  ; CHECK:   Globally Coherent: 0
  ; CHECK:   HasCounter: 0
  ; CHECK:   IsROV: 0
  ; CHECK:   Element Type: f32
  ; CHECK:   Element Count: 4

  ; CHECK-NOT: Binding {{[0-9]+}}:

  ret void
}

; CHECK-DAG: Call bound to [[SRV0]]: %srv0 =
; CHECK-DAG: Call bound to [[SRV1]]: %srv1 =
; CHECK-DAG: Call bound to [[SRV2]]: %srv2 =
; CHECK-DAG: Call bound to [[UAV0]]: %uav0 =
; CHECK-DAG: Call bound to [[UAV1]]: %uav1 =
; CHECK-DAG: Call bound to [[UAV2]]: %uav2_1 =
; CHECK-DAG: Call bound to [[UAV2]]: %uav2_2 =

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }
