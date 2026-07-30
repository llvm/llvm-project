; Make sure that llvm-as/llvm-dis properly assemble/disassemble the
; source_filename.

; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: source_filename = "C:\\path\\with\\backslashes\\test.cc"
source_filename = "C:\\path\\with\5Cbackslashes\\test.cc"
