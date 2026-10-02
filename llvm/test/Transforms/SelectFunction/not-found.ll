; RUN: opt -S -passes='select-function<fn=nonexistent>' < %s 2>&1 | FileCheck %s

; If the function doesn't exist, the pass should warn and leave the module unchanged.

; CHECK: select-function: function 'nonexistent' not found in module
; CHECK: define {{.*}} @foo(
define i32 @foo(i32 %x) {
  ret i32 %x
}
