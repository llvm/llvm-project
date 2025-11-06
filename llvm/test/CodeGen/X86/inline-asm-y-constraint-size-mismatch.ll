; RUN: not llc -mtriple=x86_64-unknown-linux-gnu -filetype=null %s 2>&1 | FileCheck -implicit-check-not=error %s

; Test that using MMX register constraint 'y' (64-bit) with mismatched vector sizes
; produces an error message instead of an assertion failure.
;
; This is a regression test for an assertion that would fire in getCopyFromPartsVector:
;   Assertion `RegisterVT == PartVT && "Part type doesn't match vector breakdown!"' failed
;
; The error can be caught at different stages:
; - During register allocation: "couldn't allocate output register for constraint"
; - In getCopyFromPartsVector: "register type (X) doesn't match operand type (Y)"
;
; We test multiple invalid vector sizes to ensure the fix handles various mismatches.

define <8 x i32> @test_mmx_constraint_256bit() {
; CHECK: error: {{couldn't allocate output register for constraint 'y'|register type .* doesn't match operand type}}
entry:
  %out = tail call <8 x i32> asm "something $0", "=y"()
  ret <8 x i32> %out
}

define <4 x i32> @test_mmx_constraint_128bit() {
; CHECK: error: {{couldn't allocate output register for constraint 'y'|register type .* doesn't match operand type}}
entry:
  %out = tail call <4 x i32> asm "something $0", "=y"()
  ret <4 x i32> %out
}

