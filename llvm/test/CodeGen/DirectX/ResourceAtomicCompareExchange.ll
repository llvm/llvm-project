; RUN: opt -S -dxil-resource-access -dxil-op-lower %s | FileCheck %s --check-prefixes=CHECK,I32 --implicit-check-not=insertvalue --implicit-check-not=icmp
; RUN: opt -S -dxil-resource-access -dxil-op-lower -mtriple=dxil-pc-shadermodel6.6-compute %s | FileCheck %s --check-prefixes=CHECK,I32,I64 --implicit-check-not=insertvalue --implicit-check-not=icmp

; Verify cmpxchg through a dx.resource.getpointer is lowered to
; dx.op.atomicCompareExchange for UAV resources. The DXIL op returns only the
; original value, so the users of the { value, success } pair read that value
; directly and the pair is never rebuilt.

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK-LABEL: define i32 @cmpxchg_i32(
define i32 @cmpxchg_i32(i32 %index, i32 %cmp, i32 %value) {
  %buffer = call target("dx.RawBuffer", i32, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i32, 1, 0, 0) %buffer, i32 %index)

  ; I32: [[ORIG:%.*]] = call i32 @dx.op.atomicCompareExchange.i32(i32 79, %dx.types.Handle %{{.*}}, i32 %index, i32 0, i32 poison, i32 %cmp, i32 %value)
  ; I32: ret i32 [[ORIG]]
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

  ; I32: [[ORIG:%.*]] = call i32 @dx.op.atomicCompareExchange.i32(i32 79, %dx.types.Handle %{{.*}}, i32 %offset, i32 poison, i32 poison, i32 %cmp, i32 %value)
  ; I32: ret i32 [[ORIG]]
  %pair = cmpxchg ptr %ptr, i32 %cmp, i32 %value monotonic monotonic
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

; CHECK-LABEL: define i64 @cmpxchg_i64(
define i64 @cmpxchg_i64(i32 %index, i64 %cmp, i64 %value) {
  %buffer = call target("dx.RawBuffer", i64, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 2, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i64, 1, 0, 0) %buffer, i32 %index)

  ; I64: [[ORIG:%.*]] = call i64 @dx.op.atomicCompareExchange.i64(i32 79, %dx.types.Handle %{{.*}}, i32 %index, i32 0, i32 poison, i64 %cmp, i64 %value)
  ; I64: ret i64 [[ORIG]]
  %pair = cmpxchg ptr %ptr, i64 %cmp, i64 %value monotonic monotonic
  %old = extractvalue { i64, i1 } %pair, 0
  ret i64 %old
}
