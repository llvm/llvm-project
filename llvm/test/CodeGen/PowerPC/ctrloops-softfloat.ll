; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-linux-gnu -O1 < %s | FileCheck %s

; double x, y;
; 
; void foo1()
; {
;   x = y = 1.1;
;   for (int i = 0; i < 175; i++)
;     y = x + y;    
; }
; void foo2()
; {
;   x = y = 1.1;
;   for (int i = 0; i < 175; i++)
;     y = x - y;    
; }
; void foo3()
; {
;   x = y = 1.1;
;   for (int i = 0; i < 175; i++)
;     y = x * y;    
; }
; void foo4()
; {
;   x = y = 1.1;
;   for (int i = 0; i < 175; i++)
;     y = x / y;    
; }

target datalayout = "E-m:e-p:32:32-i64:64-n32"
target triple = "powerpc-buildroot-linux-gnu"

@y = common global double 0.000000e+00, align 8
@x = common global double 0.000000e+00, align 8

define void @foo1() #0 {
  store double 1.100000e+00, ptr @y, align 8
  store double 1.100000e+00, ptr @x, align 8
  br label %2

; <label>:1                                       ; preds = %2
  %.lcssa = phi double [ %4, %2 ]
  store double %.lcssa, ptr @y, align 8
  ret void

; <label>:2                                       ; preds = %2, %0
  %3 = phi double [ 1.100000e+00, %0 ], [ %4, %2 ]
  %i.01 = phi i32 [ 0, %0 ], [ %5, %2 ]
  %4 = fadd double %3, 1.100000e+00
  %5 = add nuw nsw i32 %i.01, 1
  %exitcond = icmp eq i32 %5, 75
  br i1 %exitcond, label %1, label %2
  ; CHECK: bl __adddf3
  ; CHECK: cmplwi
  ; CHECK-NOT: li {{[0-9]+}}, 175
  ; CHECK-NOT: mtctr {{[0-9]+}}
}

define void @foo2() #0 {
  store double 1.100000e+00, ptr @y, align 8
  store double 1.100000e+00, ptr @x, align 8
  br label %2

; <label>:1                                       ; preds = %2
  %.lcssa = phi double [ %4, %2 ]
  store double %.lcssa, ptr @y, align 8
  ret void

; <label>:2                                       ; preds = %2, %0
  %3 = phi double [ 1.100000e+00, %0 ], [ %4, %2 ]
  %i.01 = phi i32 [ 0, %0 ], [ %5, %2 ]
  %4 = fsub double 1.100000e+00, %3
  %5 = add nuw nsw i32 %i.01, 1
  %exitcond = icmp eq i32 %5, 75
  br i1 %exitcond, label %1, label %2
  ; CHECK: bl __subdf3
  ; CHECK: cmplwi
  ; CHECK-NOT: li {{[0-9]+}}, 175
  ; CHECK-NOT: mtctr {{[0-9]+}}
}

define void @foo3() #0 {
  store double 1.100000e+00, ptr @y, align 8
  store double 1.100000e+00, ptr @x, align 8
  br label %2

; <label>:1                                       ; preds = %2
  %.lcssa = phi double [ %4, %2 ]
  store double %.lcssa, ptr @y, align 8
  ret void

; <label>:2                                       ; preds = %2, %0
  %3 = phi double [ 1.100000e+00, %0 ], [ %4, %2 ]
  %i.01 = phi i32 [ 0, %0 ], [ %5, %2 ]
  %4 = fmul double %3, 1.100000e+00
  %5 = add nuw nsw i32 %i.01, 1
  %exitcond = icmp eq i32 %5, 75
  br i1 %exitcond, label %1, label %2
  ; CHECK: bl __muldf3
  ; CHECK: cmplwi
  ; CHECK-NOT: li {{[0-9]+}}, 175
  ; CHECK-NOT: mtctr {{[0-9]+}}
}

define void @foo4() #0 {
  store double 1.100000e+00, ptr @y, align 8
  store double 1.100000e+00, ptr @x, align 8
  br label %2

; <label>:1                                       ; preds = %2
  %.lcssa = phi double [ %4, %2 ]
  store double %.lcssa, ptr @y, align 8
  ret void

; <label>:2                                       ; preds = %2, %0
  %3 = phi double [ 1.100000e+00, %0 ], [ %4, %2 ]
  %i.01 = phi i32 [ 0, %0 ], [ %5, %2 ]
  %4 = fdiv double 1.100000e+00, %3
  %5 = add nuw nsw i32 %i.01, 1
  %exitcond = icmp eq i32 %5, 75
  br i1 %exitcond, label %1, label %2
  ; CHECK: bl __divdf3
  ; CHECK: cmplwi
  ; CHECK-NOT: li {{[0-9]+}}, 175
  ; CHECK-NOT: mtctr {{[0-9]+}}
}

attributes #0 = { "use-soft-float"="true" }

