
; Make sure that llvm-as/llvm-dis properly assemble/disassemble the
; source_filename.

; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: source_filename = "test.cc"
source_filename = "test.cc"
