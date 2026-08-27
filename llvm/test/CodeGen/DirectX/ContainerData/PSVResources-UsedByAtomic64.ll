; RUN: llc %s -disable-dxil-remove-unused-resources --filetype=obj -o - | obj2yaml | FileCheck %s

; Two UAVs: `AtomicUAV` is used by a 64-bit atomicrmw, `PlainUAV` is only
; loaded/stored (no atomics). Only `AtomicUAV` should have the
; `UsedByAtomic64` PSV resource flag set.

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK: Resources:

; RWByteAddressBuffer AtomicUAV : register(u0);
; CHECK:        - Type:            UAVRaw
; CHECK:          Space:           0
; CHECK:          LowerBound:      0
; CHECK:          UpperBound:      0
; CHECK:          Kind:            RawBuffer
; CHECK:          Flags:
; CHECK:            UsedByAtomic64:  true

; RWBuffer<int> PlainUAV : register(u1);
; CHECK:        - Type:            UAVTyped
; CHECK:          Space:           0
; CHECK:          LowerBound:      1
; CHECK:          UpperBound:      1
; CHECK:          Kind:            TypedBuffer
; CHECK:          Flags:
; CHECK:            UsedByAtomic64:  false

define void @main() #0 {
  %atomic = call target("dx.RawBuffer", i8, 1, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)
  %atomicPtr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i8, 1, 0) %atomic, i32 0)
  %old = atomicrmw add ptr %atomicPtr, i64 1 monotonic

  %plain = call target("dx.TypedBuffer", i32, 1, 0, 1)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0_1t(
          i32 0, i32 1, i32 1, i32 0, ptr null)
  %plainPtr = call ptr @llvm.dx.resource.getpointer(
      target("dx.TypedBuffer", i32, 1, 0, 1) %plain, i32 0)
  store i32 42, ptr %plainPtr

  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
