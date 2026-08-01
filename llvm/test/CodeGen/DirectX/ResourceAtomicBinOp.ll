; RUN: opt -S -dxil-resource-access -dxil-op-lower %s | FileCheck %s --check-prefixes=CHECK,I32
; RUN: opt -S -dxil-resource-access -dxil-op-lower -mtriple=dxil-pc-shadermodel6.6-compute %s | FileCheck %s --check-prefixes=CHECK,I32,I64

; Verify atomicrmw through a dx.resource.getpointer is lowered to
; dx.op.atomicBinOp for UAV resources.

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK-LABEL: define i32 @atomic_i32(
define i32 @atomic_i32(i32 %index, i32 %value) {
  %buffer = call target("dx.RawBuffer", i32, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i32, 1, 0, 0) %buffer, i32 %index)

  ; I32: call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle %{{.*}}, i32 0, i32 %index, i32 0, i32 0, i32 %value)
  %add = atomicrmw add ptr %ptr, i32 %value monotonic
  ; I32: call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle %{{.*}}, i32 1, i32 %index, i32 0, i32 0, i32 %value)
  %and = atomicrmw and ptr %ptr, i32 %value monotonic
  ; I32: call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle %{{.*}}, i32 2, i32 %index, i32 0, i32 0, i32 %value)
  %or = atomicrmw or ptr %ptr, i32 %value monotonic
  ; I32: call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle %{{.*}}, i32 3, i32 %index, i32 0, i32 0, i32 %value)
  %xor = atomicrmw xor ptr %ptr, i32 %value monotonic
  ; I32: call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle %{{.*}}, i32 4, i32 %index, i32 0, i32 0, i32 %value)
  %min = atomicrmw min ptr %ptr, i32 %value monotonic
  ; I32: call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle %{{.*}}, i32 5, i32 %index, i32 0, i32 0, i32 %value)
  %max = atomicrmw max ptr %ptr, i32 %value monotonic
  ; I32: call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle %{{.*}}, i32 6, i32 %index, i32 0, i32 0, i32 %value)
  %umin = atomicrmw umin ptr %ptr, i32 %value monotonic
  ; I32: call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle %{{.*}}, i32 7, i32 %index, i32 0, i32 0, i32 %value)
  %umax = atomicrmw umax ptr %ptr, i32 %value monotonic
  ; I32: call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle %{{.*}}, i32 8, i32 %index, i32 0, i32 0, i32 %value)
  %xchg = atomicrmw xchg ptr %ptr, i32 %value monotonic
  ret i32 %xchg
}

; CHECK-LABEL: define i32 @atomic_i32_byteaddress(
define i32 @atomic_i32_byteaddress(i32 %offset, i32 %value) {
  %buffer = call target("dx.RawBuffer", i8, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i8, 1, 0, 0) %buffer, i32 %offset)

  ; I32: call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle %{{.*}}, i32 0, i32 %offset, i32 poison, i32 0, i32 %value)
  %old = atomicrmw add ptr %ptr, i32 %value monotonic
  ret i32 %old
}

; CHECK-LABEL: define i64 @atomic_i64(
define i64 @atomic_i64(i32 %index, i64 %value) {
  %buffer = call target("dx.RawBuffer", i64, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 2, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i64, 1, 0, 0) %buffer, i32 %index)

  ; I64: call i64 @dx.op.atomicBinOp.i64(i32 78, %dx.types.Handle %{{.*}}, i32 0, i32 %index, i32 0, i32 0, i64 %value)
  %old = atomicrmw add ptr %ptr, i64 %value monotonic
  ret i64 %old
}
