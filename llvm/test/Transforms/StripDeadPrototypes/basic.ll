; RUN: opt -strip-dead-prototypes -S -o - < %s | FileCheck %s
; RUN: opt -S -passes=strip-dead-prototypes < %s | FileCheck %s

; CHECK: declare i32 @f
declare i32 @f()
; CHECK-NOT: declare i32 @g
declare i32 @g()

define i32 @foo() {
  %call = call i32 @f()
  ret i32 %call
}
