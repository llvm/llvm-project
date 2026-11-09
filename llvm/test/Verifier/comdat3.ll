; This used to be invalid, but now it's valid.  Ensure the verifier
; doesn't reject it.
; RUN: llvm-as %s -disable-output

$v = comdat largest
