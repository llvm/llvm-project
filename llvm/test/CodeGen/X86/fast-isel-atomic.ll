; RUN: llc < %s -O0 -mtriple=x86_64--
; rdar://8204072
; PR7652

@sc = external global i8
@uc = external global i8

define void @test_fetch_and_op() nounwind {
entry:
  %tmp40 = atomicrmw and i8* @sc, i8 11 monotonic
  store i8 %tmp40, i8* @sc
  %tmp41 = atomicrmw and i8* @uc, i8 11 monotonic
  store i8 %tmp41, i8* @uc
  ret void
}
