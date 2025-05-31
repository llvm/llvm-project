; RUN: opt < %s -S -passes=instrprof | FileCheck %s
; RUN: opt < %s -S -passes=instrprof -runtime-counter-relocation | FileCheck -check-prefixes=RELOC %s
; RUN: opt < %s -S -passes=instrprof,inline,gvn -runtime-counter-relocation | FileCheck -check-prefixes=RELOC,RELOCOPT %s
; RUN: opt < %s -S -passes=instrprof            -runtime-counter-relocation -instrprof-atomic-counter-update-all | FileCheck -check-prefixes=ATOMIC %s
; RUN: opt < %s -S -passes=instrprof,inline,gvn -runtime-counter-relocation -instrprof-atomic-counter-update-all | FileCheck -check-prefixes=ATOMIC,ATOMICOPT %s

target triple = "x86_64-unknown-linux-gnu"

@__profn_foo = private constant [3 x i8] c"foo"
@__profn_bar = private constant [3 x i8] c"bar"
; RELOC: $__llvm_profile_counter_bias = comdat any
; RELOC: @__llvm_profile_counter_bias = linkonce_odr hidden global i64 0, comdat

; CHECK-LABEL: define void @foo
; CHECK-NEXT: %[[PGOCOUNT:.+]] = load i64, ptr @__profc_foo
; CHECK-NEXT: %[[PGOCOUNTINC:.+]] = add i64 %[[PGOCOUNT]], 1
; CHECK-NEXT: store i64 %[[PGOCOUNTINC]], ptr @__profc_foo
; RELOC-LABEL: define void @foo
; RELOC-NEXT: %[[BIAS:.+]] = load i64, ptr @__llvm_profile_counter_bias, align {{[0-9]+}}, !invariant.load !0
; RELOC-NEXT: %[[PROFC_BIAS:.+]] = add i64 ptrtoint (ptr @__profc_foo to i64), %[[BIAS]]
; RELOC-NEXT: %[[PROFC_ADDR:.+]] = inttoptr i64 %[[PROFC_BIAS]] to ptr
; RELOC-NEXT: %[[PGOCOUNT:.+]] = load i64, ptr %[[PROFC_ADDR]]
; RELOC-NEXT: %[[PGOCOUNTINC:.+]] = add i64 %[[PGOCOUNT]], 1
; RELOC-NEXT: store i64 %[[PGOCOUNTINC]], ptr %[[PROFC_ADDR]]
; RELOCOPT-NEXT: %[[PROFC_BIAS1:.+]] = add i64 ptrtoint (ptr @__profc_bar to i64), %[[BIAS]]
; RELOCOPT-NEXT: %[[PROFC_ADDR1:.+]] = inttoptr i64 %[[PROFC_BIAS1]] to ptr
; RELOCOPT-NEXT: %[[PGOCOUNT1:.+]] = load i64, ptr %[[PROFC_ADDR1]]
; RELOCOPT-NEXT: %[[PGOCOUNTINC1:.+]] = add i64 %[[PGOCOUNT1]], 1
; RELOCOPT-NEXT: store i64 %[[PGOCOUNTINC1]], ptr %[[PROFC_ADDR1]]
; ATOMIC-LABEL: define void @foo
; ATOMIC-NEXT: %[[BIAS:.+]] = load i64, ptr @__llvm_profile_counter_bias, align {{[0-9]+}}, !invariant.load !0
; ATOMIC-NEXT: %[[PROFC_BIAS:.+]] = add i64 ptrtoint (ptr @__profc_foo to i64), %[[BIAS]]
; ATOMIC-NEXT: %[[PROFC_ADDR:.+]] = inttoptr i64 %[[PROFC_BIAS]] to ptr
; ATOMIC-NEXT: %[[PGOCOUNTINC:.+]] = atomicrmw add ptr %[[PROFC_ADDR]], i64 1 monotonic
; ATOMICOPT-NEXT: %[[PROFC_BIAS1:.+]] = add i64 ptrtoint (ptr @__profc_bar to i64), %[[BIAS]]
; ATOMICOPT-NEXT: %[[PROFC_ADDR1:.+]] = inttoptr i64 %[[PROFC_BIAS1]] to ptr
; ATOMICOPT-NEXT: %[[PGOCOUNTINC1:.+]] = atomicrmw add ptr %[[PROFC_ADDR1]], i64 1 monotonic

define void @bar() {
  call void @llvm.instrprof.increment(ptr @__profn_bar, i64 0, i32 1, i32 0)
  ret void
}

define void @foo() {
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 1, i32 0)
  call void @bar()
  ret void
}

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)
