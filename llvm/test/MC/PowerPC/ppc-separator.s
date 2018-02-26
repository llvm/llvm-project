; RUN: llvm-mc -triple powerpc-apple-darwin -show-encoding -o - %s | FileCheck %s
; RUN: llvm-mc -triple powerpc64-apple-darwin -show-encoding -o - %s | FileCheck %s

_label:
	li r0, 0 @ li r1, 1

; CHECK: _label:
; CHECK: li r0, 0 ; encoding
; CHECK: li r1, 1 ; encoding

