; RUN: opt -S -hotcoldsplit -hotcoldsplit-threshold=0 < %s | FileCheck %s

; Source:
; 
; extern __attribute__((cold)) void sink();
; extern void sideeffect(int);
; void foo(int cond1, int cond2) {
;     if (cond1) {
;         if (cond2) {
;             sideeffect(0);
;         } else {
;             sideeffect(1);
;         }
;         sink();
;     } else {
;         sideeffect(2);
;     }
;     sink();
; }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; CHECK: define {{.*}}@_Z3fooii{{.*}}#[[outlined_func_attr:[0-9]+]]
; CHECK-NOT: _Z3fooii.cold
; CHECK: attributes #[[outlined_func_attr]] = { {{.*}}minsize
define void @_Z3fooii(i32, i32) {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  store i32 %1, i32* %4, align 4
  %5 = load i32, i32* %3, align 4
  %6 = icmp ne i32 %5, 0
  br i1 %6, label %7, label %13

; <label>:7:                                      ; preds = %2
  %8 = load i32, i32* %4, align 4
  %9 = icmp ne i32 %8, 0
  br i1 %9, label %10, label %11

; <label>:10:                                     ; preds = %7
  call void @_Z10sideeffecti(i32 0)
  br label %12

; <label>:11:                                     ; preds = %7
  call void @_Z10sideeffecti(i32 1)
  br label %12

; <label>:12:                                     ; preds = %11, %10
  call void @_Z4sinkv() #3
  br label %14

; <label>:13:                                     ; preds = %2
  call void @_Z10sideeffecti(i32 2)
  br label %14

; <label>:14:                                     ; preds = %13, %12
  call void @_Z4sinkv() #3
  ret void
}

declare void @_Z10sideeffecti(i32)

declare void @_Z4sinkv() cold
