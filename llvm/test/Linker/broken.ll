; RUN: not llvm-link -disable-debug-info-type-map -o /dev/null %s 2>&1 | FileCheck %s

; CHECK: broken.ll: error: input module is broken!
define i32 @foo(i32 %v) {
  %first = add i32 %v, %second
  %second = add i32 %v, 3
  ret i32 %first
}
