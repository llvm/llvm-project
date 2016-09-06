; RUN: llc -march=hexagon < %s | FileCheck %s

; No arrays in sdata.
; CHECK: memb(##foo)

@foo = common global [4 x i8] zeroinitializer, align 1

define void @set() nounwind {
entry:
  store i8 0, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @foo, i32 0, i32 0), align 1
  ret void
}

