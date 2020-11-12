; RUN: opt < %s -dfsan -dfsan-track-select-control-flow=1 -S | FileCheck %s --check-prefix=TRACK_CONTROL_FLOW
; RUN: opt < %s -dfsan -dfsan-track-select-control-flow=0 -S | FileCheck %s --check-prefix=NO_TRACK_CONTROL_FLOW
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i8 @select8(i1 %c, i8 %t, i8 %f) {
  ; TRACK_CONTROL_FLOW: %1 = load i16, i16* getelementptr inbounds ([64 x i16], [64 x i16]* @__dfsan_arg_tls, i64 0, i64 2)
  ; TRACK_CONTROL_FLOW: %2 = load i16, i16* getelementptr inbounds ([64 x i16], [64 x i16]* @__dfsan_arg_tls, i64 0, i64 1)
  ; TRACK_CONTROL_FLOW: %3 = load i16, i16* getelementptr inbounds ([64 x i16], [64 x i16]* @__dfsan_arg_tls, i64 0, i64 0)
  ; TRACK_CONTROL_FLOW: %4 = select i1 %c, i16 %2, i16 %1
  ; TRACK_CONTROL_FLOW: %5 = icmp ne i16 %3, %4
  ; TRACK_CONTROL_FLOW: %7 = call {{.*}} i16 @__dfsan_union(i16 {{.*}} %3, i16 {{.*}} %4)
  ; TRACK_CONTROL_FLOW: %9 = phi i16 [ %7, {{.*}} ], [ %3, {{.*}} ]
  ; TRACK_CONTROL_FLOW: %a = select i1 %c, i8 %t, i8 %f
  ; TRACK_CONTROL_FLOW: store i16 %9, i16* @__dfsan_retval_tls
  ; TRACK_CONTROL_FLOW: ret i8 %a
  
  ; NO_TRACK_CONTROL_FLOW: %1 = load i16, i16* getelementptr inbounds ([64 x i16], [64 x i16]* @__dfsan_arg_tls, i64 0, i64 2)
  ; NO_TRACK_CONTROL_FLOW: %2 = load i16, i16* getelementptr inbounds ([64 x i16], [64 x i16]* @__dfsan_arg_tls, i64 0, i64 1)
  ; NO_TRACK_CONTROL_FLOW: %3 = load i16, i16* getelementptr inbounds ([64 x i16], [64 x i16]* @__dfsan_arg_tls, i64 0, i64 0)
  ; NO_TRACK_CONTROL_FLOW: %4 = select i1 %c, i16 %2, i16 %1
  ; NO_TRACK_CONTROL_FLOW: %a = select i1 %c, i8 %t, i8 %f
  ; NO_TRACK_CONTROL_FLOW: store i16 %4, i16* @__dfsan_retval_tls
  ; NO_TRACK_CONTROL_FLOW: ret i8 %a

  %a = select i1 %c, i8 %t, i8 %f
  ret i8 %a
}

define i8 @select8e(i1 %c, i8 %tf) {
  ; TRACK_CONTROL_FLOW: %1 = load i16, i16* getelementptr inbounds ([64 x i16], [64 x i16]* @__dfsan_arg_tls, i64 0, i64 1)
  ; TRACK_CONTROL_FLOW: %2 = load i16, i16* getelementptr inbounds ([64 x i16], [64 x i16]* @__dfsan_arg_tls, i64 0, i64 0)
  ; TRACK_CONTROL_FLOW: %3 = icmp ne i16 %2, %1
  ; TRACK_CONTROL_FLOW: %5 = call {{.*}} i16 @__dfsan_union(i16 {{.*}} %2, i16 {{.*}} %1)
  ; TRACK_CONTROL_FLOW: %7 = phi i16 [ %5, {{.*}} ], [ %2, {{.*}} ]
  ; TRACK_CONTROL_FLOW: %a = select i1 %c, i8 %tf, i8 %tf
  ; TRACK_CONTROL_FLOW: store i16 %7, i16* @__dfsan_retval_tls
  ; TRACK_CONTROL_FLOW: ret i8 %a
  
  ; NO_TRACK_CONTROL_FLOW: %1 = load i16, i16* getelementptr inbounds ([64 x i16], [64 x i16]* @__dfsan_arg_tls, i64 0, i64 1)
  ; NO_TRACK_CONTROL_FLOW: %2 = load i16, i16* getelementptr inbounds ([64 x i16], [64 x i16]* @__dfsan_arg_tls, i64 0, i64 0)
  ; NO_TRACK_CONTROL_FLOW: %a = select i1 %c, i8 %tf, i8 %tf
  ; NO_TRACK_CONTROL_FLOW: store i16 %1, i16* @__dfsan_retval_tls
  ; NO_TRACK_CONTROL_FLOW: ret i8 %a

  %a = select i1 %c, i8 %tf, i8 %tf
  ret i8 %a
}

define <4 x i8> @select8v(<4 x i1> %c, <4 x i8> %t, <4 x i8> %f) {
  ; TRACK_CONTROL_FLOW: %1 = load i16, i16* getelementptr inbounds ([64 x i16], [64 x i16]* @__dfsan_arg_tls, i64 0, i64 2)
  ; TRACK_CONTROL_FLOW: %2 = load i16, i16* getelementptr inbounds ([64 x i16], [64 x i16]* @__dfsan_arg_tls, i64 0, i64 1)
  ; TRACK_CONTROL_FLOW: %3 = load i16, i16* getelementptr inbounds ([64 x i16], [64 x i16]* @__dfsan_arg_tls, i64 0, i64 0)
  ; TRACK_CONTROL_FLOW: %4 = icmp ne i16 %2, %1
  ; TRACK_CONTROL_FLOW: %6 = call {{.*}} i16 @__dfsan_union(i16 {{.*}} %2, i16 zeroext %1)
  ; TRACK_CONTROL_FLOW: %8 = phi i16 [ %6, {{.*}} ], [ %2, {{.*}} ]
  ; TRACK_CONTROL_FLOW: %9 = icmp ne i16 %3, %8
  ; TRACK_CONTROL_FLOW: %11 = call {{.*}} i16 @__dfsan_union(i16 {{.*}} %3, i16 zeroext %8)
  ; TRACK_CONTROL_FLOW: %13 = phi i16 [ %11, {{.*}} ], [ %3, {{.*}} ]
  ; TRACK_CONTROL_FLOW: %a = select <4 x i1> %c, <4 x i8> %t, <4 x i8> %f
  ; TRACK_CONTROL_FLOW: store i16 %13, i16* @__dfsan_retval_tls
  ; TRACK_CONTROL_FLOW: ret <4 x i8> %a

  ; NO_TRACK_CONTROL_FLOW: %1 = load i16, i16* getelementptr inbounds ([64 x i16], [64 x i16]* @__dfsan_arg_tls, i64 0, i64 2)
  ; NO_TRACK_CONTROL_FLOW: %2 = load i16, i16* getelementptr inbounds ([64 x i16], [64 x i16]* @__dfsan_arg_tls, i64 0, i64 1)
  ; NO_TRACK_CONTROL_FLOW: %3 = load i16, i16* getelementptr inbounds ([64 x i16], [64 x i16]* @__dfsan_arg_tls, i64 0, i64 0)
  ; NO_TRACK_CONTROL_FLOW: %4 = icmp ne i16 %2, %1
  ; NO_TRACK_CONTROL_FLOW: %6 = call {{.*}} i16 @__dfsan_union(i16 {{.*}} %2, i16 {{.*}} %1)
  ; NO_TRACK_CONTROL_FLOW: %8 = phi i16 [ %6, {{.*}} ], [ %2, {{.*}} ]
  ; NO_TRACK_CONTROL_FLOW: %a = select <4 x i1> %c, <4 x i8> %t, <4 x i8> %f
  ; NO_TRACK_CONTROL_FLOW: store i16 %8, i16* @__dfsan_retval_tls
  ; NO_TRACK_CONTROL_FLOW: ret <4 x i8> %a

  %a = select <4 x i1> %c, <4 x i8> %t, <4 x i8> %f
  ret <4 x i8> %a
}