; RUN: llc -asm-verbose=false < %s -wasm-keep-registers | FileCheck %s

; Test that 128-bit smul.with.overflow does not emit a call to __muloti4
; when the function has the "no-builtins" attribute. This avoids infinite
; recursion when the function is a custom implementation of __muloti4.
; See https://github.com/llvm/llvm-project/issues/189173

target triple = "wasm32-unknown-unknown"

define i128 @custom_muloti4(i128 %a, i128 %b) nounwind "no-builtins" {
entry:
  %smul = tail call { i128, i1 } @llvm.smul.with.overflow.i128(i128 %a, i128 %b)
  %cmp = extractvalue { i128, i1 } %smul, 1
  %smul.result = extractvalue { i128, i1 } %smul, 0
  %X = select i1 %cmp, i128 %smul.result, i128 42
  ret i128 %X
}

; CHECK-NOT: call __muloti4

declare { i128, i1 } @llvm.smul.with.overflow.i128(i128, i128) nounwind readnone
