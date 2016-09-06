
; Make sure that llvm-as/llvm-dis properly assemble/disassemble the
; source_filename.

; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: source_filename = "C:\5Cpath\5Cwith\5Cbackslashes\5Ctest.cc"
source_filename = "C:\5Cpath\5Cwith\5Cbackslashes\5Ctest.cc"
