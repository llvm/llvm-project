; RUN: not --crash llc -O0 -mtriple=spirv64-unknown-unknown -filetype=null %s 2>&1 | FileCheck %s

; SPIR-V has no stack guard global, so getSDagStackGuard() returns null when
; the IRTranslator emits the stack protector check. Check that the
; IRTranslator reports a translation failure instead of dereferencing the
; null value.

; CHECK: error: unable to lower stackguard
; CHECK: LLVM ERROR: unable to translate basic block (in function: test_sspreq)
define void @test_sspreq() sspreq {
  %buf = alloca [16 x i8], align 1
  store volatile i8 0, ptr %buf
  ret void
}
