; RUN: llc < %s -O0 -mtriple=x86_64--
; rdar://8204072
; PR7652

@sc = external global i8
@uc = external global i8

define void @test_fetch_and_op() nounwind {
entry:
  %tmp40 = atomicrmw and ptr @sc, i8 11 monotonic
  store i8 %tmp40, ptr @sc
  %tmp41 = atomicrmw and ptr @uc, i8 11 monotonic
  store i8 %tmp41, ptr @uc
  ret void
}
