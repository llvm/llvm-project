; RUN: opt -S -dxil-resource-access -dxil-op-lower %s | FileCheck %s --check-prefixes=CHECK,I32
; RUN: opt -S -dxil-resource-access -dxil-op-lower -mtriple=dxil-pc-shadermodel6.6-compute %s | FileCheck %s --check-prefixes=CHECK,I32,I64

; Verify cmpxchg through a dx.resource.getpointer is lowered to
; dx.op.atomicCompareExchange for UAV resources. The DXIL op returns only the
; original value, so the { value, success } pair that cmpxchg produces is
; rebuilt by comparing the returned value against the expected one.

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK-LABEL: define i32 @cmpxchg_i32(
define i32 @cmpxchg_i32(i32 %index, i32 %cmp, i32 %value) {
  %buffer = call target("dx.RawBuffer", i32, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i32, 1, 0, 0) %buffer, i32 %index)

  ; I32: [[ORIG:%.*]] = call i32 @dx.op.atomicCompareExchange.i32(i32 79, %dx.types.Handle %{{.*}}, i32 %index, i32 0, i32 0, i32 %cmp, i32 %value)
  ; I32: [[OK:%.*]] = icmp eq i32 [[ORIG]], %cmp
  ; I32: [[AGG:%.*]] = insertvalue { i32, i1 } poison, i32 [[ORIG]], 0
  ; I32: insertvalue { i32, i1 } [[AGG]], i1 [[OK]], 1
  %pair = cmpxchg ptr %ptr, i32 %cmp, i32 %value monotonic monotonic
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

; A ByteAddressBuffer is not a struct, so the byte offset is folded into the
; index and coord1 is poison.
; CHECK-LABEL: define i32 @cmpxchg_i32_byteaddress(
define i32 @cmpxchg_i32_byteaddress(i32 %offset, i32 %cmp, i32 %value) {
  %buffer = call target("dx.RawBuffer", i8, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i8, 1, 0, 0) %buffer, i32 %offset)

  ; I32: [[ORIG:%.*]] = call i32 @dx.op.atomicCompareExchange.i32(i32 79, %dx.types.Handle %{{.*}}, i32 %offset, i32 poison, i32 0, i32 %cmp, i32 %value)
  ; I32: [[OK:%.*]] = icmp eq i32 [[ORIG]], %cmp
  %pair = cmpxchg ptr %ptr, i32 %cmp, i32 %value monotonic monotonic
  %old = extractvalue { i32, i1 } %pair, 0
  %ok = extractvalue { i32, i1 } %pair, 1
  %sel = select i1 %ok, i32 %old, i32 0
  ret i32 %sel
}

; CHECK-LABEL: define i64 @cmpxchg_i64(
define i64 @cmpxchg_i64(i32 %index, i64 %cmp, i64 %value) {
  %buffer = call target("dx.RawBuffer", i64, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 2, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i64, 1, 0, 0) %buffer, i32 %index)

  ; I64: [[ORIG:%.*]] = call i64 @dx.op.atomicCompareExchange.i64(i32 79, %dx.types.Handle %{{.*}}, i32 %index, i32 0, i32 0, i64 %cmp, i64 %value)
  ; I64: icmp eq i64 [[ORIG]], %cmp
  %pair = cmpxchg ptr %ptr, i64 %cmp, i64 %value monotonic monotonic
  %old = extractvalue { i64, i1 } %pair, 0
  ret i64 %old
}
