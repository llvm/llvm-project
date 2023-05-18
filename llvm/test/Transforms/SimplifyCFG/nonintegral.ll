; RUN: opt -passes=simplifycfg -S < %s | FileCheck %s

target datalayout = "ni:1"

define void @test_01(ptr addrspace(1) align 8 %ptr) {
; CHECK-LABEL: @test_01(
; CHECK-NOT:   ptrtoint
; CHECK-NEXT:  icmp eq ptr addrspace(1) %ptr, null
; CHECK-NOT:   ptrtoint
  %cond1 = icmp eq ptr addrspace(1) %ptr, null
  %cond2 = icmp eq ptr addrspace(1) %ptr, null
  br i1 %cond1, label %true1, label %false1

true1:
  br i1 %cond2, label %true2, label %false2

false1:
  store i64 1, ptr addrspace(1) %ptr, align 8
  br label %true1

true2:
  store i64 2, ptr addrspace(1) %ptr, align 8
  ret void

false2:
  store i64 3, ptr addrspace(1) %ptr, align 8
  ret void
}
