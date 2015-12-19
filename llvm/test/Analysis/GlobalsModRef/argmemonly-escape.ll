; RUN: opt < %s -O1 -S -enable-non-lto-gmr=true | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

@a = internal global [3 x i32] zeroinitializer, align 4

; The important thing we're checking for here is the reload of (some element of)
; @a after the memset.

; CHECK-LABEL: @main
; CHECK: load i32, i32* getelementptr {{.*}} @a
; CHECK-NEXT: call void @memsetp0i8i64{{.*}} @a
; CHECK-NEXT: load i32, i32* getelementptr {{.*}} @a
; CHECK-NEXT: call void @memsetp0i8i64A{{.*}} @a
; CHECK-NEXT: load i32, i32* getelementptr {{.*}} @a
; CHECK: icmp eq
; CHECK: br i1

define i32 @main() {
entry:
  %0 = bitcast [3 x i32]* @a to i8*
  %1 = load i32, i32* getelementptr inbounds ([3 x i32], [3 x i32]* @a, i64 0, i64 2), align 4
  call void @memsetp0i8i64(i8* %0, i8 0, i64 4, i32 4, i1 false)
  %2 = load i32, i32* getelementptr inbounds ([3 x i32], [3 x i32]* @a, i64 0, i64 2), align 4
  call void @memsetp0i8i64A(i8* %0, i8 0, i64 4, i32 4, i1 false)
  %3 = load i32, i32* getelementptr inbounds ([3 x i32], [3 x i32]* @a, i64 0, i64 2), align 4
  %4 = add i32 %2, %3
  %cmp1 = icmp eq i32 %1, %4
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %entr
  call void @abort() #3
  unreachable

if.end:                                           ; preds = %entry
  ret i32 0
}

; Function Attrs: nounwind argmemonly
declare void @memsetp0i8i64(i8* nocapture, i8, i64, i32, i1) nounwind argmemonly

; Function Attrs: nounwind inaccessiblemem_or_argmemonly
declare void @memsetp0i8i64A(i8* nocapture, i8, i64, i32, i1) nounwind inaccessiblemem_or_argmemonly

; Function Attrs: noreturn nounwind
declare void @abort() noreturn nounwind
