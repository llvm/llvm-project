; RUN: llc < %s -mtriple=sparcv9 | FileCheck %s

define dso_local zeroext i32 @f() local_unnamed_addr {
entry:
  %0 = tail call i64 asm "", "=r"()
  %shr = lshr i64 %0, 32
  %conv = trunc i64 %shr to i32
  ret i32 %conv
}
; CHECK: srlx
