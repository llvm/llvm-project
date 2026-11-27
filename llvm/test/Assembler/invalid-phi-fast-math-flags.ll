; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: error: fast-math-flags specified for phi without floating-point scalar or vector return type
define i32 @f(i32 %v) {
entry:
  br label %b
b:
  %p = phi nnan i32 [ %v, %entry ]
  ret i32 %p
}
