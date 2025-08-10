; RUN: llc < %s -mtriple=riscv64 -O3 | FileCheck %s --check-prefix=RV64

declare i32 @llvm.ct.select.i32(i1, i32, i32)

define i32 @test_ctselect_i32(i1 %cond, i32 %a, i32 %b) {
; RV64-LABEL: test_ctselect_i32:
; RV64:       # %bb.0:
; RV64-NEXT:    andi
; RV64-NEXT:    addi
; RV64-NEXT:    neg
; RV64-NEXT:    and
; RV64-NEXT:    and
; RV64-NEXT:    or
; RV64-NEXT:    ret
  %r = call i32 @llvm.ct.select.i32(i1 %cond, i32 %a, i32 %b)
  ret i32 %r
}
