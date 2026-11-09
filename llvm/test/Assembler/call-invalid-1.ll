; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

declare void @f()

define void @g() {
  call void @f() align 8
; CHECK: Attribute 'align 8' does not apply to functions!
  ret void
}
