; RUN: opt < %s -passes=sccp -S | FileCheck %s

define i1 @test_cmpxchg(i32* %addr, i32 %desired, i32 %new) {
; CHECK-LABEL: @test_cmpxchg
; CHECK: cmpxchg i32* %addr, i32 %desired, i32 %new seq_cst seq_cst
  %val = cmpxchg i32* %addr, i32 %desired, i32 %new seq_cst seq_cst
  %res = extractvalue { i32, i1 } %val, 1
  ret i1 %res
}
