; RUN: opt < %s -passes=instcombine -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-pc-win32"

declare void @use(i32) readonly

; We prefer to canonicalize the machine width gep indices early
define void @test(ptr %p, i32 %index) {
; CHECK-LABEL: @test
; CHECK-NEXT: %1 = sext i32 %index to i64
; CHECK-NEXT: %addr = getelementptr i32, ptr %p, i64 %1
  %addr = getelementptr i32, ptr %p, i32 %index
  %val = load i32, ptr %addr
  call void @use(i32 %val)
  ret void
}
; If they've already been canonicalized via zext, that's fine
define void @test2(ptr %p, i32 %index) {
; CHECK-LABEL: @test2
; CHECK-NEXT: %i = zext i32 %index to i64
; CHECK-NEXT: %addr = getelementptr i32, ptr %p, i64 %i
  %i = zext i32 %index to i64
  %addr = getelementptr i32, ptr %p, i64 %i
  %val = load i32, ptr %addr
  call void @use(i32 %val)
  ret void
}
; If we can use a zext, we prefer that.  This requires
; knowing that the index is positive.
define void @test3(ptr %p, i32 %index) {
; CHECK-LABEL: @test3
; CHECK:   zext
; CHECK-NOT: sext
  %addr_begin = getelementptr i32, ptr %p, i64 40
  %addr_fixed = getelementptr i32, ptr %addr_begin, i64 48
  %val_fixed = load i32, ptr %addr_fixed, !range !0
  %addr = getelementptr i32, ptr %addr_begin, i32 %val_fixed
  %val = load i32, ptr %addr
  call void @use(i32 %val)
  ret void
}
; Replace sext with zext where possible
define void @test4(ptr %p, i32 %index) {
; CHECK-LABEL: @test4
; CHECK:   zext
; CHECK-NOT: sext
  %addr_begin = getelementptr i32, ptr %p, i64 40
  %addr_fixed = getelementptr i32, ptr %addr_begin, i64 48
  %val_fixed = load i32, ptr %addr_fixed, !range !0
  %i = sext i32 %val_fixed to i64
  %addr = getelementptr i32, ptr %addr_begin, i64 %i
  %val = load i32, ptr %addr
  call void @use(i32 %val)
  ret void
}

;;  !range !0
!0 = !{i32 0, i32 2147483647}



