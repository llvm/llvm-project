; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

; Make sure resource table is created correctly.
; DXC: Resources:
target triple = "dxil-unknown-shadermodel6.0-compute"

define void @main() #0 {

  ; ByteAddressBuffer Buf : register(t8, space1)
; DXC:        - Type:            SRVRaw
; DXC:          Space:           1
; DXC:          LowerBound:      8
; DXC:          UpperBound:      8
; DXC:          Kind:            RawBuffer
; DXC:          Flags:
; DXC:            UsedByAtomic64:  false
  %srv0 = call target("dx.RawBuffer", i8, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.RawBuffer_i8_0_0t(
          i32 1, i32 8, i32 1, i32 0, i1 false)

  ; struct S { float4 a; uint4 b; };
  ; StructuredBuffer<S> Buf : register(t2, space4)
; DXC:        - Type:            SRVStructured
; DXC:          Space:           4
; DXC:          LowerBound:      2
; DXC:          UpperBound:      2
; DXC:          Kind:            StructuredBuffer
; DXC:          Flags:
; DXC:            UsedByAtomic64:  false
  %srv1 = call target("dx.RawBuffer", {<4 x float>, <4 x i32>}, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.RawBuffer_sl_v4f32v4i32s_0_0t(
          i32 4, i32 2, i32 1, i32 0, i1 false)

  ; Buffer<uint4> Buf[24] : register(t3, space5)
; DXC:        - Type:            SRVTyped
; DXC:          Space:           5
; DXC:          LowerBound:      3
; DXC:          UpperBound:      26
; DXC:          Kind:            TypedBuffer
; DXC:          Flags:
; DXC:            UsedByAtomic64:  false
  %srv2 = call target("dx.TypedBuffer", <4 x i32>, 0, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.TypedBuffer_i32_0_0t(
          i32 5, i32 3, i32 24, i32 0, i1 false)

  ; RWBuffer<int> Buf : register(u7, space2)
; DXC:        - Type:            UAVTyped
; DXC:          Space:           2
; DXC:          LowerBound:      7
; DXC:          UpperBound:      7
; DXC:          Kind:            TypedBuffer
; DXC:          Flags:
; DXC:            UsedByAtomic64:  false
  %uav0 = call target("dx.TypedBuffer", i32, 1, 0, 1)
      @llvm.dx.handle.fromBinding.tdx.TypedBuffer_i32_1_0t(
          i32 2, i32 7, i32 1, i32 0, i1 false)

  ; RWBuffer<float4> Buf : register(u5, space3)
; DXC:        - Type:            UAVTyped
; DXC:          Space:           3
; DXC:          LowerBound:      5
; DXC:          UpperBound:      5
; DXC:          Kind:            TypedBuffer
; DXC:          Flags:
; DXC:            UsedByAtomic64:  false
  %uav1 = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
              @llvm.dx.handle.fromBinding.tdx.TypedBuffer_f32_1_0(
                  i32 3, i32 5, i32 1, i32 0, i1 false)

  ; RWBuffer<float4> BufferArray[10] : register(u0, space4)
; DXC:        - Type:            UAVTyped
; DXC:          Space:           4
; DXC:          LowerBound:      0
; DXC:          UpperBound:      9
; DXC:          Kind:            TypedBuffer
; DXC:          Flags:
; DXC:            UsedByAtomic64:  false
  ; RWBuffer<float4> Buf = BufferArray[0]
  %uav2_1 = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
              @llvm.dx.handle.fromBinding.tdx.TypedBuffer_f32_1_0(
                  i32 4, i32 0, i32 10, i32 0, i1 false)
  ; RWBuffer<float4> Buf = BufferArray[5]
  %uav2_2 = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
              @llvm.dx.handle.fromBinding.tdx.TypedBuffer_f32_1_0(
                  i32 4, i32 0, i32 10, i32 5, i1 false)
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

!dx.valver = !{!0}

!0 = !{i32 1, i32 7}
