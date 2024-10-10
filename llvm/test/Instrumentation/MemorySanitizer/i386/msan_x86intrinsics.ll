; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1 | FileCheck  \
; RUN: %s
; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=1 -S          \
; RUN: -passes=msan 2>&1 | FileCheck -check-prefix=CHECK                       \
; RUN: -check-prefix=CHECK-ORIGINS %s
; REQUIRES: x86-registered-target

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32"
target triple = "i386-unknown-linux-gnu"

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
; CHECK:       [[TMP1:%.*]] = load <8 x i16>, ptr inttoptr (i32 ptrtoint (ptr @__msan_param_tls to i32) to ptr), align 8
; CHECK:       [[TMP3:%.*]] = load <8 x i16>, ptr inttoptr (i32 add (i32 ptrtoint (ptr @__msan_param_tls to i32), i32 16) to ptr), align 8
; CHECK:       [[TMP5:%.*]] = load i64, ptr @__msan_va_arg_overflow_size_tls, align 4
; CHECK:       [[TMP6:%.*]] = trunc i64 [[TMP5]] to i32
; CHECK:       [[TMP7:%.*]] = add i32 0, [[TMP6]]
; CHECK:       call void @llvm.donothing()
; CHECK:       [[_MSPROP:%.*]] = or <8 x i16> [[TMP1]], [[TMP3]]
; CHECK:       [[CALL:%.*]] = call <8 x i16> @llvm.x86.sse2.pmulhu.w
; CHECK:       store <8 x i16> [[_MSPROP]], ptr @__msan_retval_tls, align 8
; CHECK:       ret <8 x i16> [[CALL]]
