; RUN: llc < %s -march=x86 | FileCheck %s

; Don't crash on an empty struct member.

; CHECK: movl  $2, 4(%esp)
; CHECK: movl  $1, (%esp)

%testType = type {i32, [0 x i32], i32}

define void @foo() nounwind {
  %1 = alloca %testType
  store volatile %testType {i32 1, [0 x i32] zeroinitializer, i32 2}, %testType* %1
  ret void
}
