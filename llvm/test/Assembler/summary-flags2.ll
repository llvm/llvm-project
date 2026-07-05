; Check that we can assemble/disassemble empty index with non-trivial flags
; RUN: llvm-as %s -o - | llvm-dis -o - | FileCheck %s

; ModuleID = 'index.bc'
source_filename = "index.bc"

^0 = flags: 2
; CHECK: ^0 = flags: 2
