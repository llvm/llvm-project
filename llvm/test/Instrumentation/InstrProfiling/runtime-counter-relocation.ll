; RUN: opt < %s -S -passes=instrprof | FileCheck %s
; RUN: opt < %s -S -passes=instrprof -runtime-counter-relocation | FileCheck -check-prefixes=RELOC %s

target triple = "x86_64-unknown-linux-gnu"

@__profn_foo = private constant [3 x i8] c"foo"
; RELOC: $__llvm_profile_counter_bias = comdat any
; RELOC: @__llvm_profile_counter_bias = linkonce_odr hidden global i64 0, comdat

; CHECK-LABEL: define void @foo
; CHECK-NEXT: %[[PGOCOUNT:.+]] = load i64, ptr @__profc_foo
; CHECK-NEXT: %[[PGOCOUNTINC:.+]] = add i64 %[[PGOCOUNT]], 1
; CHECK-NEXT: store i64 %[[PGOCOUNTINC]], ptr @__profc_foo
; RELOC-LABEL: define void @foo
; RELOC-NEXT: %[[BIAS:.+]] = load i64, ptr @__llvm_profile_counter_bias
; RELOC-NEXT: %[[PROFC_BIAS:.+]] = add i64 ptrtoint (ptr @__profc_foo to i64), %[[BIAS]]
; RELOC-NEXT: %[[PROFC_ADDR:.+]] = inttoptr i64 %[[PROFC_BIAS]] to ptr
; RELOC-NEXT: %[[PGOCOUNT:.+]] = load i64, ptr %[[PROFC_ADDR]]
; RELOC-NEXT: %[[PGOCOUNTINC:.+]] = add i64 %[[PGOCOUNT]], 1
; RELOC-NEXT: store i64 %[[PGOCOUNTINC]], ptr %[[PROFC_ADDR]]
define void @foo() {
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)
