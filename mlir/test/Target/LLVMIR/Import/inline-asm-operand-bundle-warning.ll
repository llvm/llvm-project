; RUN: mlir-translate -import-llvm -verify-diagnostics %s | FileCheck %s

; CHECK-LABEL: llvm.func @inline_asm(
; CHECK-SAME: %[[ARG:.*]]: i32,
; CHECK: %[[RESULT:.*]] = llvm.inline_asm has_side_effects asm_dialect = att "", "=r,0" %[[ARG]] : (i32) -> i32
; CHECK-NEXT: llvm.return %[[RESULT]] : i32
define i32 @inline_asm(i32 %arg, i32 %state) {
  ; expected-warning @unknown {{dropping operand bundles on inline assembly}}
  %result = call i32 asm sideeffect "", "=r,0"(i32 %arg) ["tag"(i32 %state)]
  ret i32 %result
}
