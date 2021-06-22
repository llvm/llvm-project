; RUN: opt < %s -dfsan -dfsan-track-origins=1  -S | FileCheck %s
;
; %15 and %17 have the same key in shadow cache. They should not reuse the same
; shadow because their blocks do not dominate each other. Origin tracking
; splt blocks. This test ensures DT is updated correctly, and cached shadows
; are not mis-used.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_arg_tls = external thread_local(initialexec) global [[TLS_ARR:\[100 x i64\]]]
; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define void @cached_shadows(double %0) {
  ; CHECK: @cached_shadows.dfsan
  ; CHECK:  [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK:  [[AS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN:2]]
  ; CHECK: [[L1:[0-9]+]]:
  ; CHECK:  {{.*}} = phi i[[#SBITS]]
  ; CHECK:  {{.*}} = phi i32
  ; CHECK:  {{.*}} = phi double [ 3.000000e+00
  ; CHECK:  [[S_L1:%.*]] = phi i[[#SBITS]] [ 0, %[[L0:[0-9]+]] ], [ [[S_L7:%.*]], %[[L7:[0-9]+]] ]
  ; CHECK:  [[O_L1:%.*]] = phi i32 [ 0, %[[L0]] ], [ [[O_L7:%.*]], %[[L7]] ]
  ; CHECK:  [[V_L1:%.*]] = phi double [ 4.000000e+00, %[[L0]] ], [ [[V_L7:%.*]], %[[L7]] ]
  ; CHECK:  br i1 {{%[0-9]+}}, label %[[L2:[0-9]+]], label %[[L4:[0-9]+]]
  ; CHECK: [[L2]]:
  ; CHECK:  br i1 {{%[0-9]+}}, label %[[L3:[0-9]+]], label %[[L7]]
  ; CHECK: [[L3]]:
  ; CHECK:  [[S_L3:%.*]] = or i[[#SBITS]]
  ; CHECK:  [[AS_NE_L3:%.*]] = icmp ne i[[#SBITS]] [[AS]], 0
  ; CHECK:  [[O_L3:%.*]] = select i1 [[AS_NE_L3]], i32 %2, i32 [[O_L1]]
  ; CHECK:  [[V_L3:%.*]] = fsub double [[V_L1]], %0
  ; CHECK:  br label %[[L7]]
  ; CHECK: [[L4]]:
  ; CHECK:  br i1 %_dfscmp, label %[[L5:[0-9]+]], label %[[L6:[0-9]+]]
  ; CHECK: [[L5]]:
  ; CHECK:  br label %[[L6]]
  ; CHECK: [[L6]]:
  ; CHECK:  [[S_L6:%.*]] = or i[[#SBITS]]
  ; CHECK:  [[AS_NE_L6:%.*]] = icmp ne i[[#SBITS]] [[AS]], 0
  ; CHECK:  [[O_L6:%.*]] = select i1 [[AS_NE_L6]], i32 [[AO]], i32 [[O_L1]]
  ; CHECK:  [[V_L6:%.*]] = fadd double [[V_L1]], %0
  ; CHECK:  br label %[[L7]]
  ; CHECK: [[L7]]:
  ; CHECK:  [[S_L7]] = phi i[[#SBITS]] [ [[S_L3]], %[[L3]] ], [ [[S_L1]], %[[L2]] ], [ [[S_L6]], %[[L6]] ]
  ; CHECK:  [[O_L7]] = phi i32 [ [[O_L3]], %[[L3]] ], [ [[O_L1]], %[[L2]] ], [ [[O_L6]], %[[L6]] ]
  ; CHECK:  [[V_L7]] = phi double [ [[V_L3]], %[[L3]] ], [ [[V_L1]], %[[L2]] ], [ [[V_L6]], %[[L6]] ]
  ; CHECK:  br i1 {{%[0-9]+}}, label %[[L1]], label %[[L8:[0-9]+]]
  ; CHECK: [[L8]]:
  
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  %4 = bitcast double* %2 to i8*
  store volatile double 1.000000e+00, double* %2, align 8
  %5 = bitcast double* %3 to i8*
  store volatile double 2.000000e+00, double* %3, align 8
  br label %6

6:                                                ; preds = %18, %1
  %7 = phi double [ 3.000000e+00, %1 ], [ %19, %18 ]
  %8 = phi double [ 4.000000e+00, %1 ], [ %20, %18 ]
  %9 = load volatile double, double* %3, align 8
  %10 = fcmp une double %9, 0.000000e+00
  %11 = load volatile double, double* %3, align 8
  br i1 %10, label %12, label %16

12:                                               ; preds = %6
  %13 = fcmp une double %11, 0.000000e+00
  br i1 %13, label %14, label %18

14:                                               ; preds = %12
  %15 = fsub double %8, %0
  br label %18

16:                                               ; preds = %6
  store volatile double %11, double* %2, align 8
  %17 = fadd double %8, %0
  br label %18

18:                                               ; preds = %16, %14, %12
  %19 = phi double [ %8, %14 ], [ %7, %12 ], [ %8, %16 ]
  %20 = phi double [ %15, %14 ], [ %8, %12 ], [ %17, %16 ]
  %21 = fcmp olt double %19, 9.900000e+01
  br i1 %21, label %6, label %22

22:                                               ; preds = %18
  ret void
}