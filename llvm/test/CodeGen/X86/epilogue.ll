; RUN: llc < %s -mcpu=generic -mtriple=i686-- | FileCheck %s

; CHECK-NOT: lea{{.*}}(%esp)
; CHECK: {{(mov.* %ebp, %esp)|(lea.*\(%ebp\), %esp)}}

declare void @bar(ptr %n)

define void @foo(i64 %h) {
  %k = trunc i64 %h to i32
  %p = alloca <2 x i64>, i32 %k
  call void @bar(ptr %p)
  ret void
}
