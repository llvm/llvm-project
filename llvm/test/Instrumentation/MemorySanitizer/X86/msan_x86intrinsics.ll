; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1 | FileCheck  \
; RUN: %s
; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=1 -S          \
; RUN: -passes=msan 2>&1 | FileCheck -check-prefix=CHECK                       \
; RUN: -check-prefix=CHECK-ORIGINS %s
; REQUIRES: x86-registered-target

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Store intrinsic.

define void @StoreIntrinsic(ptr %p, <4 x float> %x) nounwind uwtable sanitize_memory {
  call void @llvm.x86.sse.storeu.ps(ptr %p, <4 x float> %x)
  ret void
}

declare void @llvm.x86.sse.storeu.ps(ptr, <4 x float>) nounwind

; CHECK-LABEL: @StoreIntrinsic
; CHECK-NOT: br
; CHECK-NOT: = or
; CHECK: store <4 x i32> {{.*}} align 1
; CHECK: store <4 x float> %{{.*}}, ptr %{{.*}}, align 1{{$}}
; CHECK: ret void


; Load intrinsic.

define <16 x i8> @LoadIntrinsic(ptr %p) nounwind uwtable sanitize_memory {
  %call = call <16 x i8> @llvm.x86.sse3.ldu.dq(ptr %p)
  ret <16 x i8> %call
}

declare <16 x i8> @llvm.x86.sse3.ldu.dq(ptr %p) nounwind

; CHECK-LABEL: @LoadIntrinsic
; CHECK: load <16 x i8>, ptr {{.*}} align 1
; CHECK-ORIGINS: [[ORIGIN:%[01-9a-z]+]] = load i32, ptr {{.*}}
; CHECK-NOT: br
; CHECK-NOT: = or
; CHECK: call <16 x i8> @llvm.x86.sse3.ldu.dq
; CHECK: store <16 x i8> {{.*}} @__msan_retval_tls
; CHECK-ORIGINS: store i32 {{.*}}[[ORIGIN]], ptr @__msan_retval_origin_tls
; CHECK: ret <16 x i8>


; Simple NoMem intrinsic
; Check that shadow is OR'ed, and origin is Select'ed
; And no shadow checks!

define <8 x i16> @Pmulhuw128(<8 x i16> %a, <8 x i16> %b) nounwind uwtable sanitize_memory {
  %call = call <8 x i16> @llvm.x86.sse2.pmulhu.w(<8 x i16> %a, <8 x i16> %b)
  ret <8 x i16> %call
}

declare <8 x i16> @llvm.x86.sse2.pmulhu.w(<8 x i16> %a, <8 x i16> %b) nounwind

; CHECK-LABEL: @Pmulhuw128
; CHECK-NEXT: load <8 x i16>, ptr @__msan_param_tls
; CHECK-ORIGINS: load i32, ptr @__msan_param_origin_tls
; CHECK-NEXT: load <8 x i16>, ptr {{.*}} @__msan_param_tls
; CHECK-ORIGINS: load i32, ptr {{.*}} @__msan_param_origin_tls
; CHECK-NEXT: call void @llvm.donothing
; CHECK-NEXT: = or <8 x i16>
; CHECK-ORIGINS: = bitcast <8 x i16> {{.*}} to i128
; CHECK-ORIGINS-NEXT: = icmp ne i128 {{.*}}, 0
; CHECK-ORIGINS-NEXT: = select i1 {{.*}}, i32 {{.*}}, i32
; CHECK-NEXT: call <8 x i16> @llvm.x86.sse2.pmulhu.w
; CHECK-NEXT: store <8 x i16> {{.*}} @__msan_retval_tls
; CHECK-ORIGINS: store i32 {{.*}} @__msan_retval_origin_tls
; CHECK-NEXT: ret <8 x i16>
