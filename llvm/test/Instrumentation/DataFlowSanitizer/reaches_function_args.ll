; RUN: opt < %s -passes=dfsan -dfsan-reaches-function-callbacks=1 -S | FileCheck %s
; RUN: opt < %s -passes=dfsan -dfsan-reaches-function-callbacks=1 -dfsan-track-origins=1 -S | FileCheck %s --check-prefix=ORIGIN
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Each argument gets one callback with its shadow from __dfsan_arg_tls. The
; inserted instrumentation must not itself be instrumented: no shadow loads for
; the __dfsan_arg_tls loads, and no __dfsan_arg_tls stores for the callbacks.

; CHECK-LABEL: define void @args.dfsan(i32 %a, i32 %b)
; CHECK-NEXT:    %[[#SB:]] = load i8, ptr getelementptr (i8, ptr @__dfsan_arg_tls, i64 2), align 2
; CHECK-NEXT:    call void @__dfsan_reaches_function_callback(i8 zeroext %[[#SB]], ptr @{{.*}}, i32 0, ptr @{{.*}})
; CHECK-NEXT:    %[[#SA:]] = load i8, ptr @__dfsan_arg_tls, align 2
; CHECK-NEXT:    call void @__dfsan_reaches_function_callback(i8 zeroext %[[#SA]], ptr @{{.*}}, i32 0, ptr @{{.*}})
; CHECK-NEXT:    ret void

; ORIGIN-LABEL: define void @args.dfsan(i32 %a, i32 %b)
; ORIGIN-NEXT:    %[[#OB:]] = load i32, ptr getelementptr inbounds (i8, ptr @__dfsan_arg_origin_tls, i64 4), align 4
; ORIGIN-NEXT:    %[[#SB:]] = load i8, ptr getelementptr (i8, ptr @__dfsan_arg_tls, i64 2), align 2
; ORIGIN-NEXT:    call void @__dfsan_reaches_function_callback_origin(i8 zeroext %[[#SB]], i32 %[[#OB]], ptr @{{.*}}, i32 0, ptr @{{.*}})
; ORIGIN-NEXT:    %[[#OA:]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
; ORIGIN-NEXT:    %[[#SA:]] = load i8, ptr @__dfsan_arg_tls, align 2
; ORIGIN-NEXT:    call void @__dfsan_reaches_function_callback_origin(i8 zeroext %[[#SA]], i32 %[[#OA]], ptr @{{.*}}, i32 0, ptr @{{.*}})
; ORIGIN-NEXT:    ret void
define void @args(i32 %a, i32 %b) {
  ret void
}
