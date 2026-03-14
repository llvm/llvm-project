; RUN: opt -S -debug-pass-manager -passes=no-op-function < %s 2>&1 | FileCheck %s

; CHECK: Running pass: NoOpFunctionPass on f (3 instructions)
; CHECK: Running pass: NoOpFunctionPass on g (1 instruction)

define i32 @f(i32 %i) {
  %a = add i32 %i, 1
  %b = add i32 %a, 1
  ret i32 %b
}

define i32 @g(i32 %i) {
  ret i32 0
}
