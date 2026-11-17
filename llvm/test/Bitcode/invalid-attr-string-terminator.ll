; Bitcode with a string attribute group entry whose kind terminator has been
; overwritten, so the kind string runs into the value characters and the value
; loop starts at the end of the record.

; RUN: not llvm-dis %s.bc -o - 2>&1 | FileCheck %s

; CHECK: error: Value string not null terminated
