; RUN: opt -S -codegenprepare -mtriple=thumbv7m -disable-complex-addr-modes=false -addr-sink-new-select=true < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

; Select between two geps with different base, same constant offset
define void @test_select_twogep_base(i32* %ptr1, i32* %ptr2, i32 %value) {
; CHECK-LABEL: @test_select_twogep_base
; CHECK-NOT: select i1 %cmp, i32* %gep1, i32* %gep2
; CHECK: select i1 %cmp, i32* %ptr1, i32* %ptr2
entry:
  %cmp = icmp sgt i32 %value, 0
  %gep1 = getelementptr inbounds i32, i32* %ptr1, i32 1
  %gep2 = getelementptr inbounds i32, i32* %ptr2, i32 1
  %select = select i1 %cmp, i32* %gep1, i32* %gep2
  store i32 %value, i32* %select, align 4
  ret void
}

