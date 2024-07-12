; RUN: opt < %s -S -passes='always-inline' -non-global-value-max-name-size=5 | opt -non-global-value-max-name-size=5 -passes=verify -disable-output

; Opt should not generate too long name for labels during inlining.

define internal i32 @inner(i32 %flag) alwaysinline {
entry:
  %icmp = icmp slt i32 %flag, 0
  br i1 %icmp, label %one, label %two

one:
  ret i32 42

two:
  ret i32 44
}

define i32 @outer(i32 %x) {
entry:
  %call1 = call i32 @inner(i32 %x)
  %call2 = call i32 @inner(i32 %x)
  %ret = add i32 %call1, %call2
  ret i32 %ret
}