; Generate bitcode files with summary, as well as minimized bitcode without
; the debug metadata for the thin link.
; RUN: opt -thinlto-bc -thin-link-bitcode-file=%t.thinlink.bc -o %t.bc %s
; RUN: llvm-dis -o - %t.bc | FileCheck %s
; RUN: llvm-bcanalyzer -dump %t.bc | FileCheck --check-prefix=BCA %s

; Make sure the combined index files produced by both the normal and the
; thin link bitcode files are identical
; RUN: llvm-lto -thinlto -o %t3 %t.bc
; Copy the minimized bitcode to the regular bitcode path so the module
; paths in the index are the same (save and restore the regular bitcode
; for use again further down).
; RUN: mv %t.bc %t.bc.sv
; RUN: cp %t.thinlink.bc %t.bc
; RUN: llvm-lto -thinlto -o %t4 %t.bc
; RUN: mv %t.bc.sv %t.bc
; RUN: diff %t3.thinlto.bc %t4.thinlto.bc

; Try again using -thinlto-action to produce combined index
; RUN: rm -f %t3.thinlto.bc %t4.thinlto.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t3.thinlto.bc %t.bc
; Copy the minimized bitcode to the regular bitcode path so the module
; paths in the index are the same.
; RUN: cp %t.thinlink.bc %t.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t4.thinlto.bc %t.bc
; RUN: diff %t3.thinlto.bc %t4.thinlto.bc

; BCA: <GLOBALVAL_SUMMARY_BLOCK

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; CHECK: @g = global i8 42
@g = global i8 42

; CHECK: define void @f()
define void @f() {
  ret void
}
