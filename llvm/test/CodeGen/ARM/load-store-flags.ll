; RUN: llc -mtriple=thumbv7-apple-ios7.0 -o - %s -verify-machineinstrs | FileCheck %s

; The base register for the store is killed by the last instruction, but is
; actually also used during as part of the store itself. If an extra ADD is
; inserted, it should not kill the base.
define void @test_base_kill(i32 %v0, i32 %v1, ptr %addr) {
; CHECK-LABEL: test_base_kill:
; CHECK: adds [[NEWBASE:r[0-9]+]], r2, #4
; CHECK: stm [[NEWBASE]]!, {r0, r1, r2}

  %addr.1 = getelementptr i32, ptr %addr, i32 1
  store i32 %v0, ptr %addr.1

  %addr.2 = getelementptr i32, ptr %addr, i32 2
  store i32 %v1, ptr %addr.2

  %addr.3 = getelementptr i32, ptr %addr, i32 3
  %val = ptrtoint ptr %addr to i32
  store i32 %val, ptr %addr.3

  ret void
}

; Similar, but it's not sufficient to look at just the last instruction (where
; liveness of the base is determined). An intervening instruction might be moved
; past it to form the STM.
define void @test_base_kill_mid(i32 %v0, ptr %addr, i32 %v1) {
; CHECK-LABEL: test_base_kill_mid:
; CHECK: adds [[NEWBASE:r[0-9]+]], r1, #4
; CHECK: stm [[NEWBASE]]!, {r0, r1, r2}

  %addr.1 = getelementptr i32, ptr %addr, i32 1
  store i32 %v0, ptr %addr.1

  %addr.2 = getelementptr i32, ptr %addr, i32 2
  %val = ptrtoint ptr %addr to i32
  store i32 %val, ptr %addr.2

  %addr.3 = getelementptr i32, ptr %addr, i32 3
  store i32 %v1, ptr %addr.3

  ret void
}
