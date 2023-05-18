; RUN: llvm-extract -func foo -keep-const-init -S < %s | FileCheck %s
; RUN: llvm-extract -func foo -S < %s | FileCheck %s --check-prefix=CHECK2

; CHECK: @cv = constant i32 0
; CHECK2: @cv = external constant i32

@cv = constant i32 0

define i32 @foo() {
  %v = load i32, ptr @cv
  ret i32 %v
}
