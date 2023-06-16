; RUN: mlir-translate -import-llvm %s | FileCheck %s

; CHECK: llvm.mlir.global external @foo(42 : i64) comdat(@__llvm_global_comdat::@foo) {addr_space = 0 : i32} : i64
@foo = global i64 42, comdat
; CHECK: llvm.mlir.global external @bar(42 : i64) comdat(@__llvm_global_comdat::@foo) {addr_space = 0 : i32} : i64
@bar = global i64 42, comdat($foo)

; CHECK: llvm.comdat @__llvm_global_comdat {
; CHECK-DAG: llvm.comdat_selector @foo any
$foo = comdat any
; CHECK-DAG: llvm.comdat_selector @exact exactmatch
$exact = comdat exactmatch
; CHECK-DAG: llvm.comdat_selector @largest largest
$largest = comdat largest
; CHECK-DAG: llvm.comdat_selector @nodedup nodeduplicate
$nodedup = comdat nodeduplicate
; CHECK-DAG: llvm.comdat_selector @same samesize
$same = comdat samesize

; CHECK: }
