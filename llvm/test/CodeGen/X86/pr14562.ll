; RUN: llc < %s -mtriple=i686-- | FileCheck %s

@temp1 = global i64 -77129852189294865, align 8

define void @foo() nounwind {
  %x = load i64, ptr @temp1, align 8
  %s = shl i64 %x, 32
  %t = trunc i64 %s to i32
  %z = zext i32 %t to i64
  store i64 %z, ptr @temp1, align 8
; CHECK: movl $0, {{_?}}temp1+4                                                
; CHECK: movl $0, {{_?}}temp1
  ret void
}

