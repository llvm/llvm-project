; RUN: opt -S -hotcoldsplit -hotcoldsplit-threshold=0 < %s | FileCheck %s

; Source:
;
; extern __attribute__((cold)) void sink();
; extern void sideeffect(int);
; void foo(int cond1, int cond2) {
;     while (true) {
;         if (cond1) {
;             sideeffect(0); // This is cold (it reaches sink()).
;             break;
;         }
;         if (cond2) {
;             sideeffect(1); // This is cold (it reaches sink()).
;             break;
;         }
;         sideeffect(2);
;         return;
;     }
;     sink();
;     sideeffect(3);
; }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; CHECK-LABEL: define {{.*}}@_Z3fooii.cold.1
; CHECK: call void @_Z10sideeffecti(i32 1)

; CHECK-LABEL: define {{.*}}@_Z3fooii.cold.2
; CHECK: call void @_Z10sideeffecti(i32 0)

; CHECK-LABEL: define {{.*}}@_Z3fooii.cold.3
; CHECK: call void @_Z4sinkv
; CHECK: call void @_Z10sideeffecti(i32 3)

define void @_Z3fooii(i32, i32) {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  store i32 %1, ptr %4, align 4
  br label %5

; <label>:5:                                      ; preds = %2
  %6 = load i32, ptr %3, align 4
  %7 = icmp ne i32 %6, 0
  br i1 %7, label %8, label %9

; <label>:8:                                      ; preds = %5
  call void @_Z10sideeffecti(i32 0)
  br label %14

; <label>:9:                                      ; preds = %5
  %10 = load i32, ptr %4, align 4
  %11 = icmp ne i32 %10, 0
  br i1 %11, label %12, label %13

; <label>:12:                                     ; preds = %9
  call void @_Z10sideeffecti(i32 1)
  br label %14

; <label>:13:                                     ; preds = %9
  call void @_Z10sideeffecti(i32 2)
  br label %15

; <label>:14:                                     ; preds = %12, %8
  call void @_Z4sinkv() #3
  call void @_Z10sideeffecti(i32 3)
  br label %15

; <label>:15:                                     ; preds = %14, %13
  ret void
}

declare void @_Z10sideeffecti(i32)

declare void @_Z4sinkv() cold
