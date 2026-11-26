; Test that printable string data is emitted as HLASM character constants on
; z/OS: CE for EBCDIC-1047 text and CA for ASCII text, XL otherwise.

; RUN: llc < %s -mtriple=s390x-ibm-zos -asm-verbose=0 | FileCheck %s

; "TEST" in EBCDIC-1047.
@ebcdic = global [4 x i8] c"\E3\C5\E2\E3", align 1
; "it's&ok" in ASCII, with the apostrophe and ampersand doubled.
@ascii = global [7 x i8] c"it's&ok", align 1
; Non-printable data is still emitted as hex.
@binary = global [3 x i8] c"\01\02\03", align 1
; Integers are not text, even when the byte is printable.
@int = global i8 65, align 1

; CHECK-LABEL: ebcdic XATTR
; CHECK: DC CE'TEST'
; CHECK-LABEL: ascii XATTR
; CHECK: DC CA'it''s&&ok'
; CHECK-LABEL: binary XATTR
; CHECK: DC XL3'010203'
; CHECK-LABEL: int XATTR
; CHECK: DC XL1'41'
