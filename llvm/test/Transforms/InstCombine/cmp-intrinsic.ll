; RUN: opt < %s -instcombine -S | FileCheck %s

declare i16 @llvm.bswap.i16(i16)
declare i32 @llvm.bswap.i32(i32)
declare <2 x i64> @llvm.bswap.v2i64(<2 x i64>)

define i1 @bswap_eq_i16(i16 %x) {
; CHECK-LABEL: @bswap_eq_i16(
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i16 %x, 256
; CHECK-NEXT:    ret i1 [[CMP]]
;
  %bs = call i16 @llvm.bswap.i16(i16 %x)
  %cmp = icmp eq i16 %bs, 1
  ret i1 %cmp
}

define i1 @bswap_ne_i32(i32 %x) {
; CHECK-LABEL: @bswap_ne_i32(
; CHECK-NEXT:    [[CMP:%.*]] = icmp ne i32 %x, 33554432
; CHECK-NEXT:    ret i1 [[CMP]]
;
  %bs = tail call i32 @llvm.bswap.i32(i32 %x)
  %cmp = icmp ne i32 %bs, 2
  ret i1 %cmp
}

define <2 x i1> @bswap_eq_v2i64(<2 x i64> %x) {
; CHECK-LABEL: @bswap_eq_v2i64(
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq <2 x i64> %x, <i64 216172782113783808, i64 216172782113783808>
; CHECK-NEXT:    ret <2 x i1> [[CMP]]
;
  %bs = tail call <2 x i64> @llvm.bswap.v2i64(<2 x i64> %x)
  %cmp = icmp eq <2 x i64> %bs, <i64 3, i64 3>
  ret <2 x i1> %cmp
}

