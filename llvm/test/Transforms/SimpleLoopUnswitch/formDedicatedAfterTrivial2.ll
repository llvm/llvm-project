; RUN: opt < %s -simple-loop-unswitch -disable-output

; PR38283
; PR38737
define void @Test(i32) {
entry:
  %trunc = trunc i32 %0 to i3
  br label %outer
outer:
  br label %inner
inner:
  switch i3 %trunc, label %crit_edge [
    i3 2, label %break
    i3 1, label %loopexit
  ]
crit_edge:
  br i1 true, label %loopexit, label %inner
loopexit:
  ret void
break:
  br label %outer
}
