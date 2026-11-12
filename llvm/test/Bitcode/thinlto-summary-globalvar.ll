; RUN: opt -module-summary %s -o - | llvm-bcanalyzer -dump | FileCheck %s
; Check with new pass manager (by enabling a random pass in the new pipeline).
; RUN: opt -passes=gvn -module-summary %s -o - | llvm-bcanalyzer -dump | FileCheck %s

; CHECK: <GLOBALVAL_SUMMARY_BLOCK

@a = global i32 0
