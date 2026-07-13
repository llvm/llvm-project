; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: error: fast-math-flags specified for select without floating-point scalar or vector return type
define i32 @f(i1 %c, i32 %v) {
  %s = select nnan i1 %c, i32 %v, i32 0
  ret i32 %s
}
