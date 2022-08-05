; RUN: llvm-reduce --test FileCheck --test-arg --check-prefix=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=CHECK-FINAL %s < %t

; We cannot change the @alias to undef, because it would result in invalid IR
; (Aliasee should be either GlobalValue or ConstantExpr).

; CHECK-INTERESTINGNESS: @alias =
; CHECK-FINAL: @alias = alias void (i32), ptr @func

@alias = alias void (i32), ptr @func

; CHECK-FINAL: @func()

define void @func(i32 %arg) {
entry:
  ret void
}
