; Test that storage for allocas with disjoint lifetimes is reused with
; use-after-scope.

; RUN: opt -S -passes=hwasan %s -hwasan-use-after-scope -o - | \
; RUN:   llc -no-stack-coloring=false -o - | \
; RUN:   FileCheck %s --check-prefix=COLOR
; RUN: opt -S -passes=hwasan -hwasan-use-after-scope %s -o - | \
; RUN:   llc -no-stack-coloring=true -o - | \
; RUN:   FileCheck %s --check-prefix=NOCOLOR

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-android29"

; REQUIRES: aarch64-registered-target

; COLOR: sub	sp, sp, #240
; NOCOLOR: sub	sp, sp, #368

define i32 @myCall_w2(i32 %in) sanitize_hwaddress {
entry:
  %a = alloca [17 x ptr], align 8
  %a2 = alloca [16 x ptr], align 8
  call void @llvm.lifetime.start.p0(i64 136, ptr %a)
  %t1 = call i32 @foo(i32 %in, ptr %a)
  %t2 = call i32 @foo(i32 %in, ptr %a)
  call void @llvm.lifetime.end.p0(i64 136, ptr %a)
  call void @llvm.lifetime.start.p0(i64 128, ptr %a2)
  %t3 = call i32 @foo(i32 %in, ptr %a2)
  %t4 = call i32 @foo(i32 %in, ptr %a2)
  call void @llvm.lifetime.end.p0(i64 128, ptr %a2)
  %t5 = add i32 %t1, %t2
  %t6 = add i32 %t3, %t4
  %t7 = add i32 %t5, %t6
  ret i32 %t7
}

declare void @llvm.lifetime.start.p0(i64, ptr nocapture) nounwind

declare void @llvm.lifetime.end.p0(i64, ptr nocapture) nounwind

declare i32 @foo(i32, ptr)
