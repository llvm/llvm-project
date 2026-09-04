; RUN: opt -S -disable-output -passes="print<dxil-resources>" < %s 2>&1 | FileCheck %s

; Verifies that the DXILResourceMap analysis detects 64-bit atomic uses on
; UAV resources and sets `HasAtomic64Use` on the corresponding ResourceInfo.
;
; * `Atomic64UAV`  is used by a 64-bit `atomicrmw`  -> Has Atomic64 Use: 1
; * `Atomic32UAV`  is used by a 32-bit `atomicrmw`  -> Has Atomic64 Use: 0
; * `PlainUAV`     is only stored to (no atomics)   -> Has Atomic64 Use: 0

target triple = "dxil-pc-shadermodel6.6-compute"

define void @main() {
  ; RWByteAddressBuffer Atomic64UAV : register(u0);
  %h64 = call target("dx.RawBuffer", i8, 1, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)
  ; CHECK:      Binding:
  ; CHECK:        Binding ID: 0
  ; CHECK:        Space: 0
  ; CHECK:        Lower Bound: 0
  ; CHECK:        Size: 1
  ; CHECK:      Has Atomic64 Use: 1
  %p64 = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i8, 1, 0) %h64, i32 0)
  %old64 = atomicrmw add ptr %p64, i64 1 monotonic

  ; RWByteAddressBuffer Atomic32UAV : register(u1);
  %h32 = call target("dx.RawBuffer", i8, 1, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(
          i32 0, i32 1, i32 1, i32 0, ptr null)
  ; CHECK:      Binding:
  ; CHECK:        Binding ID: 1
  ; CHECK:        Space: 0
  ; CHECK:        Lower Bound: 1
  ; CHECK:        Size: 1
  ; CHECK:      Has Atomic64 Use: 0
  %p32 = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i8, 1, 0) %h32, i32 0)
  %old32 = atomicrmw add ptr %p32, i32 1 monotonic

  ; RWBuffer<int> PlainUAV : register(u2);
  %hp = call target("dx.TypedBuffer", i32, 1, 0, 1)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0_1t(
          i32 0, i32 2, i32 1, i32 0, ptr null)
  ; CHECK:      Binding:
  ; CHECK:        Binding ID: 2
  ; CHECK:        Space: 0
  ; CHECK:        Lower Bound: 2
  ; CHECK:        Size: 1
  ; CHECK:      Has Atomic64 Use: 0
  %pp = call ptr @llvm.dx.resource.getpointer(
      target("dx.TypedBuffer", i32, 1, 0, 1) %hp, i32 0)
  store i32 42, ptr %pp

  ret void
}
