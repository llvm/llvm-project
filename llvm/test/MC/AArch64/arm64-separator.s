; RUN: llvm-mc -triple arm64-apple-darwin -show-encoding < %s | FileCheck %s

; ARM64 uses a multi-character statement separator, "%%". Check that we lex
; it properly and recognize the multiple assembly statements on the line.

; To make sure the output assembly correctly handled the instructions,
; tell it to show encodings. That will result in the two 'mov' instructions
; being on separate lines in the output. We look for the "; encoding" string
; to verify that. For this test, we don't care what the encoding is, just that
; there is one for each 'mov' instruction.


_foo:
; CHECK: foo
; CHECK: mov x0, x1 ; encoding
; CHECK: mov x1, x0 ; encoding
	mov x0, x1 %% mov x1, x0
	ret	lr


