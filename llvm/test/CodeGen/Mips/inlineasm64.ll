; RUN: llc -march=mips64el -mcpu=mips64r2 -mattr=n64 < %s | FileCheck %s

@gl2 = external global i64
@gl1 = external global i64
@gl0 = external global i64

define void @foo1() nounwind {
entry:
; CHECK: foo1
; CHECK: daddu
  %0 = load i64* @gl1, align 8
  %1 = load i64* @gl0, align 8
  %2 = tail call i64 asm "daddu $0, $1, $2", "=r,r,r"(i64 %0, i64 %1) nounwind
  store i64 %2, i64* @gl2, align 8
  ret void
}

