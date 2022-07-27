; RUN: llc -mtriple=i386-linux-gnu %s -o - | FileCheck %s
; XFAIL: *

define i64 @test_add(ptr %addr, i64 %inc) {
; CHECK-LABEL: test_add:
; CHECK: calll __sync_fetch_and_add_8
  %old = atomicrmw add ptr %addr, i64 %inc seq_cst
  ret i64 %old
}

define i64 @test_sub(ptr %addr, i64 %inc) {
; CHECK-LABEL: test_sub:
; CHECK: calll __sync_fetch_and_sub_8
  %old = atomicrmw sub ptr %addr, i64 %inc seq_cst
  ret i64 %old
}

define i64 @test_and(ptr %andr, i64 %inc) {
; CHECK-LABEL: test_and:
; CHECK: calll __sync_fetch_and_and_8
  %old = atomicrmw and ptr %andr, i64 %inc seq_cst
  ret i64 %old
}

define i64 @test_or(ptr %orr, i64 %inc) {
; CHECK-LABEL: test_or:
; CHECK: calll __sync_fetch_and_or_8
  %old = atomicrmw or ptr %orr, i64 %inc seq_cst
  ret i64 %old
}

define i64 @test_xor(ptr %xorr, i64 %inc) {
; CHECK-LABEL: test_xor:
; CHECK: calll __sync_fetch_and_xor_8
  %old = atomicrmw xor ptr %xorr, i64 %inc seq_cst
  ret i64 %old
}

define i64 @test_nand(ptr %nandr, i64 %inc) {
; CHECK-LABEL: test_nand:
; CHECK: calll __sync_fetch_and_nand_8
  %old = atomicrmw nand ptr %nandr, i64 %inc seq_cst
  ret i64 %old
}
