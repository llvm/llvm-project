; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s

; Make sure resource table is created correctly.
; CHECK: Resources:
target triple = "dxil-unknown-shadermodel6.0-compute"

define void @main() #0 {

  ; cbuffer : register(b2, space3) { float x; }
; CHECK:        - Type:            CBV
; CHECK:          Space:           3
; CHECK:          LowerBound:      2
; CHECK:          UpperBound:      2
; CHECK:          Kind:            CBuffer
; CHECK:          Flags:
; CHECK:            UsedByAtomic64:  false
  %cbuf = call target("dx.CBuffer", target("dx.Layout", {float}, 4, 0))
      @llvm.dx.resource.handlefrombinding(i32 3, i32 2, i32 1, i32 0, ptr null)

  ; ByteAddressBuffer Buf : register(t8, space1)
; CHECK:        - Type:            SRVRaw
; CHECK:          Space:           1
; CHECK:          LowerBound:      8
; CHECK:          UpperBound:      8
; CHECK:          Kind:            RawBuffer
; CHECK:          Flags:
; CHECK:            UsedByAtomic64:  false
  %srv0 = call target("dx.RawBuffer", i8, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0t(
          i32 1, i32 8, i32 1, i32 0, ptr null)

  ; struct S { float4 a; uint4 b; };
  ; StructuredBuffer<S> Buf : register(t2, space4)
; CHECK:        - Type:            SRVStructured
; CHECK:          Space:           4
; CHECK:          LowerBound:      2
; CHECK:          UpperBound:      2
; CHECK:          Kind:            StructuredBuffer
; CHECK:          Flags:
; CHECK:            UsedByAtomic64:  false
  %srv1 = call target("dx.RawBuffer", {<4 x float>, <4 x i32>}, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_sl_v4f32v4i32s_0_0t(
          i32 4, i32 2, i32 1, i32 0, ptr null)

  ; Buffer<uint4> Buf[24] : register(t3, space5)
; CHECK:        - Type:            SRVTyped
; CHECK:          Space:           5
; CHECK:          LowerBound:      3
; CHECK:          UpperBound:      26
; CHECK:          Kind:            TypedBuffer
; CHECK:          Flags:
; CHECK:            UsedByAtomic64:  false
  %srv2 = call target("dx.TypedBuffer", <4 x i32>, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_0_0t(
          i32 5, i32 3, i32 24, i32 0, ptr null)

  ; RWBuffer<int> Buf : register(u7, space2)
; CHECK:        - Type:            UAVTyped
; CHECK:          Space:           2
; CHECK:          LowerBound:      7
; CHECK:          UpperBound:      7
; CHECK:          Kind:            TypedBuffer
; CHECK:          Flags:
; CHECK:            UsedByAtomic64:  false
  %uav0 = call target("dx.TypedBuffer", i32, 1, 0, 1)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0t(
          i32 2, i32 7, i32 1, i32 0, ptr null)

  ; RWBuffer<float4> Buf : register(u5, space3)
; CHECK:        - Type:            UAVTyped
; CHECK:          Space:           3
; CHECK:          LowerBound:      5
; CHECK:          UpperBound:      5
; CHECK:          Kind:            TypedBuffer
; CHECK:          Flags:
; CHECK:            UsedByAtomic64:  false
  %uav1 = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
              @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0(
                  i32 3, i32 5, i32 1, i32 0, ptr null)

  ; RWBuffer<float4> BufferArray[10] : register(u0, space4)
; CHECK:        - Type:            UAVTyped
; CHECK:          Space:           4
; CHECK:          LowerBound:      0
; CHECK:          UpperBound:      9
; CHECK:          Kind:            TypedBuffer
; CHECK:          Flags:
; CHECK:            UsedByAtomic64:  false
  ; RWBuffer<float4> Buf = BufferArray[0]
  %uav2_1 = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
              @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0(
                  i32 4, i32 0, i32 10, i32 0, ptr null)
  ; RWBuffer<float4> Buf = BufferArray[5]
  %uav2_2 = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
              @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0(
                  i32 4, i32 0, i32 10, i32 5, ptr null)

  ; RWBuffer<float4> UnboundedArray[] : register(u10, space5)
; CHECK:        - Type:            UAVTyped
; CHECK:          Space:           5
; CHECK:          LowerBound:      10
; CHECK:          UpperBound:      4294967295
; CHECK:          Kind:            TypedBuffer
; CHECK:          Flags:
; CHECK:            UsedByAtomic64:  false
  ; RWBuffer<float4> Buf = BufferArray[100];
  %uav3 = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
              @llvm.dx.resource.handlefrombinding(i32 5, i32 10, i32 -1, i32 100, ptr null)
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

!dx.valver = !{!0}

!0 = !{i32 1, i32 7}
