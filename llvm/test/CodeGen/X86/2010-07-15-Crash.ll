; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s -o /dev/null
; PR7653

@__FUNCTION__.1623 = external dso_local constant [4 x i8]   ; <ptr> [#uses=1]

define void @foo() nounwind {
entry:
  tail call void asm sideeffect "", "s,i,~{fpsr},~{flags}"(ptr @__FUNCTION__.1623, ptr @__FUNCTION__.1623) nounwind
  ret void
}
