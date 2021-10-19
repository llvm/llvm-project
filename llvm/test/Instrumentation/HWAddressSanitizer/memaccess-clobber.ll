; Make sure memaccess checks preceed the following reads.
;
; RUN: opt < %s -S -enable-new-pm=0 -hwasan -hwasan-use-stack-safety=0 -basic-aa -memdep -print-memdeps -analyze -mtriple aarch64-linux-android30 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android10000"

declare void @use32(i32*)

define i32 @test_alloca() sanitize_hwaddress {
entry:
  %x = alloca i32, align 4
  call void @use32(i32* nonnull %x)
  ; CHECK: Clobber from:   call void @llvm.hwasan.check.memaccess.shortgranule
  ; CHECK-NEXT: load i32, i32* %x.hwasan, align 4
  %y = load i32, i32* %x
  ; CHECK:  Clobber from:   %y = load i32, i32* %x.hwasan, align 4
  ; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* align 1 {{.*}}, i8 0, i64 1, i1 false)
  ret i32 %y
}
