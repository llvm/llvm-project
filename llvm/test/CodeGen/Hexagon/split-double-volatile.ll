; RUN: llc -mtriple=hexagon -O2 < %s | FileCheck %s

; Test coverage for HexagonSplitDoubleRegs: exercise the volatile memory
; operation path and the load/store splitting for 64-bit register operations.

; CHECK-LABEL: test_volatile_load:
; CHECK: memd
define i64 @test_volatile_load(ptr %p) {
entry:
  %val = load volatile i64, ptr %p, align 8
  ret i64 %val
}

; CHECK-LABEL: test_volatile_store:
; CHECK: memd
define void @test_volatile_store(ptr %p, i64 %val) {
entry:
  store volatile i64 %val, ptr %p, align 8
  ret void
}

; Non-volatile 64-bit loads can be split into two 32-bit loads.
; CHECK-LABEL: test_split_load:
; CHECK: memw
define i64 @test_split_load(ptr %p) {
entry:
  %val = load i64, ptr %p, align 4
  %add = add i64 %val, 1
  ret i64 %add
}

