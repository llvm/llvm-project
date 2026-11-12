; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK: llvm.comdat @__llvm_global_comdat {
; CHECK: llvm.comdat_selector @foo any
$foo = comdat any
; CHECK: }

; CHECK: llvm.mlir.global external @foo(42 : i64) comdat(@__llvm_global_comdat::@foo)
@foo = global i64 42, comdat
; CHECK: llvm.mlir.global external @bar(42 : i64) comdat(@__llvm_global_comdat::@foo)
@bar = global i64 42, comdat($foo)

; // -----

; CHECK: llvm.comdat @__llvm_global_comdat {
; CHECK: llvm.comdat_selector @foo any
$foo = comdat any
; CHECK: }


; CHECK: llvm.func @foo() comdat(@__llvm_global_comdat::@foo)
define void @foo() comdat {
  ret void
}
; CHECK: llvm.func @bar() comdat(@__llvm_global_comdat::@foo)
define void @bar() comdat($foo) {
  ret void
}

; // -----

; CHECK: llvm.comdat @__llvm_global_comdat {
; CHECK: llvm.comdat_selector @exact exactmatch
$exact = comdat exactmatch
; CHECK: llvm.comdat_selector @largest largest
$largest = comdat largest
; CHECK: llvm.comdat_selector @nodedup nodeduplicate
$nodedup = comdat nodeduplicate
; CHECK: llvm.comdat_selector @same samesize
$same = comdat samesize
; CHECK: }

@exact = global i64 42, comdat
@largest = global i64 42, comdat
@nodedup = global i64 42, comdat
@same = global i64 42, comdat

; // -----

; Verify a global comdat operation is only created if there are comdats to import.
; CHECK-NOT: llvm.comdat
; CHECK: llvm.mlir.global external @foobar
; CHECK-NOT: llvm.comdat
@foobar = global i64 42
