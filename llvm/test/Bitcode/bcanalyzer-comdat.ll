; RUN: llvm-as < %s | llvm-bcanalyzer -dump | FileCheck %s

; CHECK: <COMDAT
$comdat.any = comdat any
@comdat.any = global i32 0, comdat
