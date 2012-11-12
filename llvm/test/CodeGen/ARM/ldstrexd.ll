; RUN: llc < %s -mtriple=armv7-apple-darwin   | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-apple-darwin | FileCheck %s

%0 = type { i32, i32 }

; CHECK: f0:
; CHECK: ldrexd
define i64 @f0(i8* %p) nounwind readonly {
entry:
  %ldrexd = tail call %0 @llvm.arm.ldrexd(i8* %p)
  %0 = extractvalue %0 %ldrexd, 1
  %1 = extractvalue %0 %ldrexd, 0
  %2 = zext i32 %0 to i64
  %3 = zext i32 %1 to i64
  %shl = shl nuw i64 %2, 32
  %4 = or i64 %shl, %3
  ret i64 %4
}

; CHECK: f1:
; CHECK: strexd
define i32 @f1(i8* %ptr, i64 %val) nounwind {
entry:
  %tmp4 = trunc i64 %val to i32
  %tmp6 = lshr i64 %val, 32
  %tmp7 = trunc i64 %tmp6 to i32
  %strexd = tail call i32 @llvm.arm.strexd(i32 %tmp4, i32 %tmp7, i8* %ptr)
  ret i32 %strexd
}

declare %0 @llvm.arm.ldrexd(i8*) nounwind readonly
declare i32 @llvm.arm.strexd(i32, i32, i8*) nounwind

