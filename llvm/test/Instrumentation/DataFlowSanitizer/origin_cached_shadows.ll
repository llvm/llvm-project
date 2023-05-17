; RUN: opt < %s -passes=dfsan -dfsan-track-origins=1  -S | FileCheck %s
;
; %i13 and %i15 have the same key in shadow cache. They should not reuse the same
; shadow because their blocks do not dominate each other. Origin tracking
; splt blocks. This test ensures DT is updated correctly, and cached shadows
; are not mis-used.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_arg_tls = external thread_local(initialexec) global [[TLS_ARR:\[100 x i64\]]]
define void @cached_shadows(double %arg) {
  ; CHECK: @cached_shadows.dfsan
  ; CHECK:  [[AO:%.*]] = load i32, ptr @__dfsan_arg_origin_tls, align
  ; CHECK:  [[AS:%.*]] = load i8, ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; CHECK: [[L1:.+]]:
  ; CHECK:  {{.*}} = phi i8
  ; CHECK:  {{.*}} = phi i32
  ; CHECK:  {{.*}} = phi double [ 3.000000e+00
  ; CHECK:  [[S_L1:%.*]] = phi i8 [ 0, %[[L0:.*]] ], [ [[S_L7:%.*]], %[[L7:.*]] ]
  ; CHECK:  [[O_L1:%.*]] = phi i32 [ 0, %[[L0]] ], [ [[O_L7:%.*]], %[[L7]] ]
  ; CHECK:  [[V_L1:%.*]] = phi double [ 4.000000e+00, %[[L0]] ], [ [[V_L7:%.*]], %[[L7]] ]
  ; CHECK:  br i1 {{%.+}}, label %[[L2:.*]], label %[[L4:.*]]
  ; CHECK: [[L2]]:
  ; CHECK:  br i1 {{%.+}}, label %[[L3:.+]], label %[[L7]]
  ; CHECK: [[L3]]:
  ; CHECK:  [[S_L3:%.*]] = or i8
  ; CHECK:  [[AS_NE_L3:%.*]] = icmp ne i8 [[AS]], 0
  ; CHECK:  [[O_L3:%.*]] = select i1 [[AS_NE_L3]], i32 %{{[0-9]+}}, i32 [[O_L1]]
  ; CHECK:  [[V_L3:%.*]] = fsub double [[V_L1]], %{{.+}}
  ; CHECK:  br label %[[L7]]
  ; CHECK: [[L4]]:
  ; CHECK:  br i1 %_dfscmp, label %[[L5:.+]], label %[[L6:.+]],
  ; CHECK: [[L5]]:
  ; CHECK:  br label %[[L6]]
  ; CHECK: [[L6]]:
  ; CHECK:  [[S_L6:%.*]] = or i8
  ; CHECK:  [[AS_NE_L6:%.*]] = icmp ne i8 [[AS]], 0
  ; CHECK:  [[O_L6:%.*]] = select i1 [[AS_NE_L6]], i32 [[AO]], i32 [[O_L1]]
  ; CHECK:  [[V_L6:%.*]] = fadd double [[V_L1]], %{{.+}}
  ; CHECK:  br label %[[L7]]
  ; CHECK: [[L7]]:
  ; CHECK:  [[S_L7]] = phi i8 [ [[S_L3]], %[[L3]] ], [ [[S_L1]], %[[L2]] ], [ [[S_L6]], %[[L6]] ]
  ; CHECK:  [[O_L7]] = phi i32 [ [[O_L3]], %[[L3]] ], [ [[O_L1]], %[[L2]] ], [ [[O_L6]], %[[L6]] ]
  ; CHECK:  [[V_L7]] = phi double [ [[V_L3]], %[[L3]] ], [ [[V_L1]], %[[L2]] ], [ [[V_L6]], %[[L6]] ]
  ; CHECK:  br i1 %{{.+}}, label %[[L1]], label %[[L8:.+]]
  ; CHECK: [[L8]]:
bb:
  %i = alloca double, align 8
  %i1 = alloca double, align 8
  %i2 = bitcast ptr %i to ptr
  store volatile double 1.000000e+00, ptr %i, align 8
  %i3 = bitcast ptr %i1 to ptr
  store volatile double 2.000000e+00, ptr %i1, align 8
  br label %bb4

bb4:                                              ; preds = %bb16, %bb
  %i5 = phi double [ 3.000000e+00, %bb ], [ %i17, %bb16 ]
  %i6 = phi double [ 4.000000e+00, %bb ], [ %i18, %bb16 ]
  %i7 = load volatile double, ptr %i1, align 8
  %i8 = fcmp une double %i7, 0.000000e+00
  %i9 = load volatile double, ptr %i1, align 8
  br i1 %i8, label %bb10, label %bb14

bb10:                                             ; preds = %bb4
  %i11 = fcmp une double %i9, 0.000000e+00
  br i1 %i11, label %bb12, label %bb16

bb12:                                             ; preds = %bb10
  %i13 = fsub double %i6, %arg
  br label %bb16

bb14:                                             ; preds = %bb4
  store volatile double %i9, ptr %i, align 8
  %i15 = fadd double %i6, %arg
  br label %bb16

bb16:                                             ; preds = %bb14, %bb12, %bb10
  %i17 = phi double [ %i6, %bb12 ], [ %i5, %bb10 ], [ %i6, %bb14 ]
  %i18 = phi double [ %i13, %bb12 ], [ %i6, %bb10 ], [ %i15, %bb14 ]
  %i19 = fcmp olt double %i17, 9.900000e+01
  br i1 %i19, label %bb4, label %bb20

bb20:                                             ; preds = %bb16
  ret void
}
