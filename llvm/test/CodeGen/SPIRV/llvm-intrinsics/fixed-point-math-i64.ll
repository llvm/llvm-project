; i64 is unsupported for smul.fix/umul.fix because lowering widens to i128,
; which SPIR-V does not have.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o -
; XFAIL: *

define i64 @smulfix_i64(i64 %a, i64 %b) {
  %r = call i64 @llvm.smul.fix.i64(i64 %a, i64 %b, i32 3)
  ret i64 %r
}
