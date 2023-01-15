; RUN: llc  < %s -march=mips -mcpu=mips32r2 | FileCheck %s -check-prefix=32R2
; RUN: llc  < %s -march=mips -mattr=mips16 | FileCheck %s -check-prefix=16

define i32 @ext0_5_9(i32 %s, i32 %pos, i32 %sz) nounwind readnone {
entry:
; 32R2: ext ${{[0-9]+}}, $4, 5, 9
; 16-NOT: ext ${{[0-9]+}}
  %shr = lshr i32 %s, 5
  %and = and i32 %shr, 511
  ret i32 %and
}

define void @ins2_5_9(i32 %s, ptr nocapture %d) nounwind {
entry:
; 32R2: ins ${{[0-9]+}}, $4, 5, 9
; 16-NOT: ins ${{[0-9]+}}
  %and = shl i32 %s, 5
  %shl = and i32 %and, 16352
  %tmp3 = load i32, ptr %d, align 4
  %and5 = and i32 %tmp3, -16353
  %or = or i32 %and5, %shl
  store i32 %or, ptr %d, align 4
  ret void
}
