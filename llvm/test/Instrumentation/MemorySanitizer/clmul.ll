; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1 | FileCheck  \
; RUN: %s
; REQUIRES: x86-registered-target

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare <2 x i64> @llvm.x86.pclmulqdq(<2 x i64>, <2 x i64>, i8 immarg) nounwind readnone
declare <4 x i64> @llvm.x86.pclmulqdq.256(<4 x i64>, <4 x i64>, i8 immarg) nounwind readnone
declare <8 x i64> @llvm.x86.pclmulqdq.512(<8 x i64>, <8 x i64>, i8 immarg) nounwind readnone

define <2 x i64> @clmul00(<2 x i64> %a, <2 x i64> %b) sanitize_memory {
entry:
  %0 = tail call <2 x i64> @llvm.x86.pclmulqdq(<2 x i64> %a, <2 x i64> %b, i8 0)
  ret <2 x i64> %0
}

; CHECK-LABEL: @clmul00
; CHECK: %[[S0:.*]] = load <2 x i64>, ptr {{.*}}@__msan_param_tls
; CHECK: %[[S1:.*]] = load <2 x i64>, ptr {{.*}}@__msan_param_tls
; CHECK: %[[SHUF0:.*]] = shufflevector <2 x i64> %[[S0]], <2 x i64> poison, <2 x i32> zeroinitializer
; CHECK: %[[SHUF1:.*]] = shufflevector <2 x i64> %[[S1]], <2 x i64> poison, <2 x i32> zeroinitializer
; CHECK: %[[SRET:.*]] = or <2 x i64> %[[SHUF0]], %[[SHUF1]]
; CHECK: store <2 x i64> %[[SRET]], ptr {{.*}}@__msan_retval_tls

define <2 x i64> @clmul10(<2 x i64> %a, <2 x i64> %b) sanitize_memory {
entry:
  %0 = tail call <2 x i64> @llvm.x86.pclmulqdq(<2 x i64> %a, <2 x i64> %b, i8 16)
  ret <2 x i64> %0
}

; CHECK-LABEL: @clmul10
; CHECK: %[[S0:.*]] = load <2 x i64>, ptr {{.*}}@__msan_param_tls
; CHECK: %[[S1:.*]] = load <2 x i64>, ptr {{.*}}@__msan_param_tls
; CHECK: %[[SHUF0:.*]] = shufflevector <2 x i64> %[[S0]], <2 x i64> poison, <2 x i32> zeroinitializer
; CHECK: %[[SHUF1:.*]] = shufflevector <2 x i64> %[[S1]], <2 x i64> poison, <2 x i32> <i32 1, i32 1>
; CHECK: %[[SRET:.*]] = or <2 x i64> %[[SHUF0]], %[[SHUF1]]
; CHECK: store <2 x i64> %[[SRET]], ptr {{.*}}@__msan_retval_tls

define <4 x i64> @clmul11_256(<4 x i64> %a, <4 x i64> %b) sanitize_memory {
entry:
  %0 = tail call <4 x i64> @llvm.x86.pclmulqdq.256(<4 x i64> %a, <4 x i64> %b, i8 17)
  ret <4 x i64> %0
}

; CHECK-LABEL: @clmul11_256
; CHECK: %[[S0:.*]] = load <4 x i64>, ptr {{.*}}@__msan_param_tls
; CHECK: %[[S1:.*]] = load <4 x i64>, ptr {{.*}}@__msan_param_tls
; CHECK: %[[SHUF0:.*]] = shufflevector <4 x i64> %[[S0]], <4 x i64> poison, <4 x i32> <i32 1, i32 1, i32 3, i32 3>
; CHECK: %[[SHUF1:.*]] = shufflevector <4 x i64> %[[S1]], <4 x i64> poison, <4 x i32> <i32 1, i32 1, i32 3, i32 3>
; CHECK: %[[SRET:.*]] = or <4 x i64> %[[SHUF0]], %[[SHUF1]]
; CHECK: store <4 x i64> %[[SRET]], ptr {{.*}}@__msan_retval_tls

define <8 x i64> @clmul01_512(<8 x i64> %a, <8 x i64> %b) sanitize_memory {
entry:
  %0 = tail call <8 x i64> @llvm.x86.pclmulqdq.512(<8 x i64> %a, <8 x i64> %b, i8 16)
  ret <8 x i64> %0
}

; CHECK-LABEL: @clmul01_512
; CHECK: %[[S0:.*]] = load <8 x i64>, ptr {{.*}}@__msan_param_tls
; CHECK: %[[S1:.*]] = load <8 x i64>, ptr {{.*}}@__msan_param_tls
; CHECK: %[[SHUF0:.*]] = shufflevector <8 x i64> %[[S0]], <8 x i64> poison, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
; CHECK: %[[SHUF1:.*]] = shufflevector <8 x i64> %[[S1]], <8 x i64> poison, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
; CHECK: %[[SRET:.*]] = or <8 x i64> %[[SHUF0]], %[[SHUF1]]
; ORIGIN: %[[FLAT:.*]] = bitcast <8 x i64> %[[SHUF1]] to i512
; ORIGIN: %[[I:.*]] = icmp ne i512 %[[FLAT]], 0
; ORIGIN: %[[O:.*]] = select i1 %[[I]],
; CHECK: store <8 x i64> %[[SRET]], ptr {{.*}}@__msan_retval_tls
; ORIGIN: store i32 %[[O]], i32* @__msan_retval_origin_tls
