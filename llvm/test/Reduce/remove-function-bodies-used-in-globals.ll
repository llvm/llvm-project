; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL %s

; CHECK-INTERESTINGNESS: @alias =
; CHECK-FINAL: @alias = alias void (i32), void (i32)* undef

@alias = alias void (i32), void (i32)* @func

; CHECK-FINAL-NOT: @func()

define void @func(i32 %arg) {
entry:
  ret void
}
