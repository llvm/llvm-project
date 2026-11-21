; RUN: rm -rf %t && split-file %s %t
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mattr=+retpoline-indirect-calls < %t/ok.ll | FileCheck %t/ok.ll
; RUN: not --crash llc -mtriple=x86_64-unknown-linux-gnu -mattr=+retpoline-indirect-calls < %t/bad.ll 2>&1 | FileCheck %t/bad.ll
; RUN: not --crash llc -mtriple=x86_64-unknown-linux-gnu -mattr=+lvi-cfi < %t/bad.ll 2>&1 | FileCheck %t/bad.ll

; preserve_none passes the 11th integer argument in R11 (R11D for a 32-bit
; value). The retpoline and LVI thunks use R11 to hold the call target, so a
; call with 11 such arguments must be rejected instead of silently overwriting
; the argument. A call with 10 arguments leaves R11 free and must still work.

;--- ok.ll
; CHECK-LABEL: call_10_args:
; CHECK: callq __llvm_retpoline_r11
define i32 @call_10_args(ptr %fp) {
  %r = call preserve_nonecc i32 %fp(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10)
  ret i32 %r
}

;--- bad.ll
; CHECK: LLVM ERROR: calling convention incompatible with retpoline, no available registers
define i32 @call_11_args(ptr %fp) {
  %r = call preserve_nonecc i32 %fp(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11)
  ret i32 %r
}
