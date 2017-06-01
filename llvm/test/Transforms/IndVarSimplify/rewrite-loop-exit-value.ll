; RUN: opt -indvars -instcombine -S < %s | FileCheck %s

;; Test that loop's exit value is rewritten to its initial
;; value from loop preheader
define i32 @test1(i32* %var) {
; CHECK-LABEL: @test1
entry:
 %cond = icmp eq i32* %var, null 
 br label %header

header:
 %phi_indvar = phi i32 [0, %entry], [%indvar, %loop]
 br i1 %cond, label %loop, label %exit

loop:
 %indvar = add i32 %phi_indvar, 1
 br label %header

exit:
; CHECK: ret i32 0
 ret i32 %phi_indvar
}

;; Test that we can not rewrite loop exit value if it's not
;; a phi node (%indvar is an add instruction in this test).
define i32 @test2(i32* %var) {
; CHECK-LABEL: @test2
entry:
 %cond = icmp eq i32* %var, null 
 br label %header

header:
 %phi_indvar = phi i32 [0, %entry], [%indvar, %header]
 %indvar = add i32 %phi_indvar, 1
 br i1 %cond, label %header, label %exit

exit:
; CHECK: ret i32 %indvar
 ret i32 %indvar
}

;; Test that we can not rewrite loop exit value if the condition
;; is not in loop header.
define i32 @test3(i32* %var) {
; CHECK-LABEL: @test3
entry:
 %cond1 = icmp eq i32* %var, null 
 br label %header

header:
 %phi_indvar = phi i32 [0, %entry], [%indvar, %header], [%indvar, %body]
 %indvar = add i32 %phi_indvar, 1
 %cond2 = icmp eq i32 %indvar, 10
 br i1 %cond2, label %header, label %body
 
body:
 br i1 %cond1, label %header, label %exit

exit:
; CHECK: ret i32 %phi_indvar
 ret i32 %phi_indvar
}

