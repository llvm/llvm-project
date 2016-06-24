; RUN: opt -module-summary %s -o - | llvm-bcanalyzer -dump | FileCheck %s

; CHECK: <GLOBALVAL_SUMMARY_BLOCK

@a = global i32 0
