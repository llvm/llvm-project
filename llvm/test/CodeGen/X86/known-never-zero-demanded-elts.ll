; RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu %s -o - | FileCheck %s

; Helper: build vector with lane0 set, other lanes poison.

declare <4 x i8>  @llvm.ctpop.v4i8(<4 x i8>)
declare <4 x i16> @llvm.bswap.v4i16(<4 x i16>)
declare <4 x i8>  @llvm.bitreverse.v4i8(<4 x i8>)
declare <4 x i8>  @llvm.fshl.v4i8(<4 x i8>, <4 x i8>, <4 x i8>)
declare <4 x i8>  @llvm.fshr.v4i8(<4 x i8>, <4 x i8>, <4 x i8>)
declare <4 x i8>  @llvm.abs.v4i8(<4 x i8>, i1)

define i1 @ctpop_lane0_nonzero() {
; CHECK-LABEL: ctpop_lane0_nonzero:
; CHECK: mov{{.*}}1
; CHECK: ret
  %v0 = insertelement <4 x i8> poison, i8 8, i64 0
  %w  = call <4 x i8> @llvm.ctpop.v4i8(<4 x i8> %v0)
  %e0 = extractelement <4 x i8> %w, i64 0
  %cmp = icmp ne i8 %e0, 0
  ret i1 %cmp
}

define i1 @bswap_lane0_nonzero() {
; CHECK-LABEL: bswap_lane0_nonzero:
; CHECK: mov{{.*}}1
; CHECK: ret
  %v0 = insertelement <4 x i16> poison, i16 1, i64 0
  %w  = call <4 x i16> @llvm.bswap.v4i16(<4 x i16> %v0)
  %e0 = extractelement <4 x i16> %w, i64 0
  %cmp = icmp ne i16 %e0, 0
  ret i1 %cmp
}

define i1 @bitreverse_lane0_nonzero() {
; CHECK-LABEL: bitreverse_lane0_nonzero:
; CHECK: mov{{.*}}1
; CHECK: ret
  %v0 = insertelement <4 x i8> poison, i8 1, i64 0
  %w  = call <4 x i8> @llvm.bitreverse.v4i8(<4 x i8> %v0)
  %e0 = extractelement <4 x i8> %w, i64 0
  %cmp = icmp ne i8 %e0, 0
  ret i1 %cmp
}

define i1 @rotl_lane0_nonzero() {
; CHECK-LABEL: rotl_lane0_nonzero:
; CHECK: mov{{.*}}1
; CHECK: ret
  %x  = insertelement <4 x i8> poison, i8 2, i64 0
  %k  = insertelement <4 x i8> poison, i8 1, i64 0
  %w  = call <4 x i8> @llvm.fshl.v4i8(<4 x i8> %x, <4 x i8> %x, <4 x i8> %k)
  %e0 = extractelement <4 x i8> %w, i64 0
  %cmp = icmp ne i8 %e0, 0
  ret i1 %cmp
}

define i1 @rotr_lane0_nonzero() {
; CHECK-LABEL: rotr_lane0_nonzero:
; CHECK: mov{{.*}}1
; CHECK: ret
  %x  = insertelement <4 x i8> poison, i8 2, i64 0
  %k  = insertelement <4 x i8> poison, i8 1, i64 0
  %w  = call <4 x i8> @llvm.fshr.v4i8(<4 x i8> %x, <4 x i8> %x, <4 x i8> %k)
  %e0 = extractelement <4 x i8> %w, i64 0
  %cmp = icmp ne i8 %e0, 0
  ret i1 %cmp
}

define i1 @abs_lane0_nonzero() {
; CHECK-LABEL: abs_lane0_nonzero:
; CHECK: mov{{.*}}1
; CHECK: ret
  %v0 = insertelement <4 x i8> poison, i8 -2, i64 0
  %w  = call <4 x i8> @llvm.abs.v4i8(<4 x i8> %v0, i1 false)
  %e0 = extractelement <4 x i8> %w, i64 0
  %cmp = icmp ne i8 %e0, 0
  ret i1 %cmp
}
