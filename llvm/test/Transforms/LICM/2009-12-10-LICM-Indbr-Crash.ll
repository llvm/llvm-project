; Test for rdar://7452967
; RUN: opt < %s -licm -disable-output
define void @foo (i8* %v)
{
  entry:
    br i1 undef, label %preheader, label %return

  preheader:
    br i1 undef, label %loop, label %return

  loop:
    indirectbr i8* undef, [label %preheader, label %stuff]

  stuff:
    %0 = load i8, i8* undef, align 1
    br label %loop

  return:
    ret void

}
