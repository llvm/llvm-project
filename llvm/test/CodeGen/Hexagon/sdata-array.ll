; RUN: llc -march=hexagon < %s | FileCheck %s

; No arrays in sdata.
; CHECK: memb(##foo)

@foo = common global [4 x i8] zeroinitializer, align 1

define void @set(i8 %x) nounwind {
entry:
  store i8 %x, ptr @foo, align 1
  ret void
}

