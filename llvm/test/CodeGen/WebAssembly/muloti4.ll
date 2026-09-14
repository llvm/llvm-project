; RUN: llc -asm-verbose=false < %s -wasm-keep-registers | FileCheck %s

; Test that 128-bit smul.with.overflow is expanded inline rather than emitting
; a call to __muloti4. The MULO_I128 libcall was removed from WebAssembly to
; avoid infinite recursion when compiling custom implementations of __muloti4
; (e.g. Zig's compiler-rt).
; See https://github.com/llvm/llvm-project/issues/189173

target triple = "wasm32-unknown-unknown"

define i128 @call_muloti4(i128 %a, i128 %b) nounwind {
entry:
  %smul = tail call { i128, i1 } @llvm.smul.with.overflow.i128(i128 %a, i128 %b)
  %cmp = extractvalue { i128, i1 } %smul, 1
  %smul.result = extractvalue { i128, i1 } %smul, 0
  %X = select i1 %cmp, i128 %smul.result, i128 42
  ret i128 %X
}

; CHECK-NOT: call __muloti4

declare { i128, i1 } @llvm.smul.with.overflow.i128(i128, i128) nounwind readnone
