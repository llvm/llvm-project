; RUN: llc -march=mips -mcpu=mips32r2 < %s | FileCheck %s

define i32 @ext0_5_9(i32 %s, i32 %pos, i32 %sz) nounwind readnone {
entry:
; CHECK: ext ${{[0-9]+}}, $4, 5, 9
  %shr = lshr i32 %s, 5
  %and = and i32 %shr, 511
  ret i32 %and
}

define void @ins2_5_9(i32 %s, i32* nocapture %d) nounwind {
entry:
; CHECK: ins ${{[0-9]+}}, $4, 5, 9
  %and = shl i32 %s, 5
  %shl = and i32 %and, 16352
  %tmp3 = load i32* %d, align 4
  %and5 = and i32 %tmp3, -16353
  %or = or i32 %and5, %shl
  store i32 %or, i32* %d, align 4
  ret void
}
