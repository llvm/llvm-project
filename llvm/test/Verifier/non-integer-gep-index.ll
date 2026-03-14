; RUN: not opt -verify %s.bc -disable-output

; Test that verifier checks that gep indexes has correct type
; Specifically we want to check for the following pattern:
;   %A1 = alloca i64
;   %G = getelementptr i64, i64* %A1, %A1
; IR parser checks for this pattern independently from the verifier, so it's
; impossible to load from .ll file. Hence in this test we use bytecode input.
