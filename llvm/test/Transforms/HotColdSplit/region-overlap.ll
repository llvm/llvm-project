; RUN: opt -S -hotcoldsplit -hotcoldsplit-threshold=0 < %s | FileCheck %s

; Source:
;
; __attribute__((cold)) extern void sink(int);
; extern void sideeffect(int);
; void foo(int cond1, int cond2) {
;     if (cond1) {
;         if (cond2) { // This is the first cold region we visit.
;             sideeffect(0);
;             sink(0);
;         }
;
;         // There's a larger, overlapping cold region here. But we ignore it.
;         // This could be improved.
;         sideeffect(1);
;         sink(1);
;     }
; }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; CHECK-LABEL: define {{.*}}@_Z3fooii
; CHECK: call {{.*}}@_Z3fooii.cold.1
; CHECK-NOT: _Z3fooii.cold
define void @_Z3fooii(i32, i32) {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  store i32 %1, ptr %4, align 4
  %5 = load i32, ptr %3, align 4
  %6 = icmp ne i32 %5, 0
  br i1 %6, label %7, label %12

; <label>:7:                                      ; preds = %2
  %8 = load i32, ptr %4, align 4
  %9 = icmp ne i32 %8, 0
  br i1 %9, label %10, label %11

; <label>:10:                                     ; preds = %7
  call void @_Z10sideeffecti(i32 0)
  call void @_Z4sinki(i32 0) #3
  br label %11

; <label>:11:                                     ; preds = %10, %7
  call void @_Z10sideeffecti(i32 1)
  call void @_Z4sinki(i32 1) #3
  br label %12

; <label>:12:                                     ; preds = %11, %2
  ret void
}

; CHECK-LABEL: define {{.*}}@_Z3fooii.cold.1
; CHECK: call void @_Z10sideeffecti(i32 0)

declare void @_Z10sideeffecti(i32)

declare void @_Z4sinki(i32) cold
