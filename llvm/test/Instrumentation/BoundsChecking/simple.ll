; RUN: opt < %s -bounds-checking -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@.str = private constant [8 x i8] c"abcdefg\00"   ; <[8 x i8]*>

declare noalias i8* @malloc(i64) nounwind
declare noalias i8* @calloc(i64, i64) nounwind
declare noalias i8* @realloc(i8* nocapture, i64) nounwind

; CHECK: @f1
define void @f1() nounwind {
  %1 = tail call i8* @malloc(i64 32)
  %2 = bitcast i8* %1 to i32*
  %idx = getelementptr inbounds i32* %2, i64 2
; CHECK-NOT: trap
  store i32 3, i32* %idx, align 4
  ret void
}

; CHECK: @f2
define void @f2() nounwind {
  %1 = tail call i8* @malloc(i64 32)
  %2 = bitcast i8* %1 to i32*
  %idx = getelementptr inbounds i32* %2, i64 8
; CHECK: trap
  store i32 3, i32* %idx, align 4
  ret void
}

; CHECK: @f3
define void @f3(i64 %x) nounwind {
  %1 = tail call i8* @calloc(i64 4, i64 %x)
  %2 = bitcast i8* %1 to i32*
  %idx = getelementptr inbounds i32* %2, i64 8
; CHECK: mul i64 4, %
; CHECK: sub i64 {{.*}}, 32
; CHECK-NEXT: icmp ult i64 {{.*}}, 32
; CHECK-NEXT: icmp ult i64 {{.*}}, 4
; CHECK-NEXT: or i1
; CHECK: trap
  store i32 3, i32* %idx, align 4
  ret void
}

; CHECK: @f4
define void @f4(i64 %x) nounwind {
  %1 = tail call i8* @realloc(i8* null, i64 %x) nounwind
  %2 = bitcast i8* %1 to i32*
  %idx = getelementptr inbounds i32* %2, i64 8
; CHECK: trap
  %3 = load i32* %idx, align 4
  ret void
}

; CHECK: @f5
define void @f5(i64 %x) nounwind {
  %idx = getelementptr inbounds [8 x i8]* @.str, i64 0, i64 %x
; CHECK: trap
  %1 = load i8* %idx, align 4
  ret void
}

; CHECK: @f6
define void @f6(i64 %x) nounwind {
  %1 = alloca i128
; CHECK-NOT: trap
  %2 = load i128* %1, align 4
  ret void
}

; CHECK: @f7
define void @f7(i64 %x) nounwind {
  %1 = alloca i128, i64 %x
; CHECK: mul i64 16,
; CHECK: trap
  %2 = load i128* %1, align 4
  ret void
}

; CHECK: @f8
define void @f8() nounwind {
  %1 = alloca i128
  %2 = alloca i128
  %3 = select i1 undef, i128* %1, i128* %2
; CHECK-NOT: trap
  %4 = load i128* %3, align 4
  ret void
}

; CHECK: @f9
define void @f9(i128* %arg) nounwind {
  %1 = alloca i128
  %2 = select i1 undef, i128* %arg, i128* %1
; CHECK-NOT: trap
  %3 = load i128* %2, align 4
  ret void
}

; CHECK: @f10
define void @f10(i64 %x, i64 %y) nounwind {
  %1 = alloca i128, i64 %x
  %2 = alloca i128, i64 %y
  %3 = select i1 undef, i128* %1, i128* %2
; CHECK: select
; CHECK: select
; CHECK: trap
  %4 = load i128* %3, align 4
  ret void
}

; CHECK: @f11
define void @f11(i128* byval %x) nounwind {
  %1 = bitcast i128* %x to i8*
  %2 = getelementptr inbounds i8* %1, i64 16
; CHECK: br label
  %3 = load i8* %2, align 4
  ret void
}

; CHECK: @f12
define i64 @f12(i64 %x, i64 %y) nounwind {
  %1 = tail call i8* @calloc(i64 1, i64 %x)
; CHECK: mul i64 %y, 8
  %2 = bitcast i8* %1 to i64*
  %3 = getelementptr inbounds i64* %2, i64 %y
  %4 = load i64* %3, align 8
  ret i64 %4
}
