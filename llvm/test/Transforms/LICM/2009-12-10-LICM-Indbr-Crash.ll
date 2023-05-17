; Test for rdar://7452967
; RUN: opt < %s -passes=licm -disable-output
define void @foo (ptr %v)
{
  entry:
    br i1 undef, label %preheader, label %return

  preheader:
    br i1 undef, label %loop, label %return

  loop:
    indirectbr ptr undef, [label %preheader, label %stuff]

  stuff:
    %0 = load i8, ptr undef, align 1
    br label %loop

  return:
    ret void

}
