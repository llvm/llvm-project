; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; Tests bug: 24640
; CHECK: expected '=' in global variable

@- 0xKate potb8ed
