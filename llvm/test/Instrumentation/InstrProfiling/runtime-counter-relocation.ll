; RUN: opt < %s -S -instrprof | FileCheck %s
; RUN: opt < %s -S -instrprof -runtime-counter-relocation | FileCheck -check-prefixes=RELOC %s

target triple = "x86_64-unknown-linux-gnu"

@__profn_foo = hidden constant [3 x i8] c"foo"
; RELOC: @__llvm_profile_counter_bias = linkonce_odr hidden global i64 0

; CHECK-LABEL: define void @foo
; CHECK-NEXT: %pgocount = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_foo, i64 0, i64 0)
; CHECK-NEXT: %1 = add i64 %pgocount, 1
; CHECK-NEXT: store i64 %1, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_foo, i64 0, i64 0)
; RELOC-LABEL: define void @foo
; RELOC-NEXT: %1 = load i64, i64* @__llvm_profile_counter_bias
; RELOC-NEXT: %2 = add i64 ptrtoint ([1 x i64]* @__profc_foo to i64), %1
; RELOC-NEXT: %3 = inttoptr i64 %2 to i64*
; RELOC-NEXT: %pgocount = load i64, i64* %3
; RELOC-NEXT: %4 = add i64 %pgocount, 1
; RELOC-NEXT: store i64 %4, i64* %3
define void @foo() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(i8*, i64, i32, i32)
