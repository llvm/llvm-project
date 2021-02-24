;; Ensure that SHT_NOBITS section type is set for __llvm_prf_cnts in ELF.
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s

@__profc_foo = hidden global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8

define void @foo() {
  %pgocount = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_foo, i64 0, i64 0), align 4
  %1 = add i64 %pgocount, 1
  store i64 %1, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_foo, i64 0, i64 0), align 4
  ret void
}

declare void @llvm.instrprof.increment(i8*, i64, i32, i32)

; CHECK: .section __llvm_prf_cnts,"aw",@nobits
