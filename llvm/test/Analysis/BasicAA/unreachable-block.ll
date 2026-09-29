; RUN: opt -aa-pipeline=basic-aa -passes=aa-eval -disable-output < %s > /dev/null 2>&1

; BasicAA shouldn't infinitely recurse on the use-def cycles in
; unreachable code.

define void @func_2() nounwind {
entry:
  unreachable

bb:
  %t = select i1 undef, ptr %t, ptr undef
  %p = select i1 undef, ptr %p, ptr %p
  %q = select i1 undef, ptr undef, ptr %p
  unreachable
}
