; RUN: opt < %s -dfsan -dfsan-track-origins=1  -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_arg_tls = external thread_local(initialexec) global [[TLS_ARR:\[100 x i64\]]]
; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define i32 @phiop(i32 %a, i32 %b, i1 %c) {
  ; CHECK: @phiop.dfsan
  ; CHECK: entry:
  ; CHECK: [[BO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; CHECK: [[BS:%.*]] = load i[[#SBITS]], ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align [[ALIGN:2]]
  ; CHECK: [[AS:%.*]] = load i[[#SBITS]], ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; CHECK: br i1 %c, label %next, label %done
  ; CHECK: next:
  ; CHECK: br i1 %c, label %T, label %F
  ; CHECK: T:
  ; CHECK: [[BS_NE:%.*]] = icmp ne i[[#SBITS]] [[BS]], 0
  ; CHECK: [[BAO_T:%.*]] = select i1 [[BS_NE]], i32 [[BO]], i32 [[AO]]
  ; CHECK: br label %done
  ; CHECK: F:
  ; CHECK: [[AS_NE:%.*]] = icmp ne i[[#SBITS]] [[AS]], 0
  ; CHECK: [[BAO_F:%.*]] = select i1 [[AS_NE]], i32 [[AO]], i32 [[BO]]
  ; CHECK: br label %done
  ; CHECK: done:
  ; CHECK: [[PO:%.*]] = phi i32 [ [[BAO_T]], %T ], [ [[BAO_F]], %F ], [ [[AO]], %entry ]
  ; CHECK: store i32 [[PO]], ptr @__dfsan_retval_origin_tls, align 4

entry:
  br i1 %c, label %next, label %done
next:  
  br i1 %c, label %T, label %F 
T:
  %sum = add i32 %a, %b 
  br label %done
F:
  %diff = sub i32 %b, %a 
  br label %done
done:
  %r = phi i32 [%sum, %T], [%diff, %F], [%a, %entry]
  ret i32 %r
}