; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s -o /dev/null
; PR7653

@__FUNCTION__.1623 = external dso_local constant [4 x i8]   ; <[4 x i8]*> [#uses=1]

define void @foo() nounwind {
entry:
  tail call void asm sideeffect "", "s,i,~{fpsr},~{flags}"(i8* getelementptr
inbounds ([4 x i8], [4 x i8]* @__FUNCTION__.1623, i64 0, i64 0), i8* getelementptr
inbounds ([4 x i8], [4 x i8]* @__FUNCTION__.1623, i64 0, i64 0)) nounwind
  ret void
}
