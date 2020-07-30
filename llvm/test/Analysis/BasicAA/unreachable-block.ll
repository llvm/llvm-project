; RUN: opt -basic-aa -aa-eval -disable-output < %s > /dev/null 2>&1

; BasicAA shouldn't infinitely recurse on the use-def cycles in
; unreachable code.

define void @func_2() nounwind {
entry:
  unreachable

bb:
  %t = select i1 undef, i32* %t, i32* undef
  %p = select i1 undef, i32* %p, i32* %p
  %q = select i1 undef, i32* undef, i32* %p
  %a = getelementptr i8, i8* %a, i32 0
  unreachable
}
