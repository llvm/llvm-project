; RUN: llc -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu -verify-machineinstrs -enable-branch-coalesce=true < %s | FileCheck %s 
; RUN: llc -mcpu=pwr8 -mtriple=powerpc64-unknown-linux-gnu -verify-machineinstrs -enable-branch-coalesce=true < %s | FileCheck %s 

; Function Attrs: nounwind
define double @testBranchCoal(double %a, double %b, double %c, i32 %x) {
entry:
  %test = icmp eq i32 %x, 0
  %tmp1 = select i1 %test, double %a, double 2.000000e-03
  %tmp2 = select i1 %test, double %b, double 0.000000e+00
  %tmp3 = select i1 %test, double %c, double 5.000000e-03

  %res1 = fadd double %tmp1, %tmp2
  %result = fadd double %res1, %tmp3
  ret double %result

; CHECK-LABEL: @testBranchCoal 
; CHECK: cmplwi [[CMPR:[0-7]+]], 6, 0
; CHECK: beq [[CMPR]], .LBB[[LAB1:[0-9_]+]]
; CHECK-DAG: addis [[LD1REG:[0-9]+]], 2, .LCPI0_0@toc@ha
; CHECK-DAG: addis [[LD2REG:[0-9]+]], 2, .LCPI0_1@toc@ha
; CHECK-DAG: xxlxor 2, 2, 2
; CHECK-NOT: beq 
; CHECK-DAG: addi [[LD1BASE:[0-9]+]], [[LD1REG]] 
; CHECK-DAG: addi [[LD2BASE:[0-9]+]], [[LD2REG]]
; CHECK-DAG: lxsdx 1, 0, [[LD1BASE]]
; CHECK-DAG: lxsdx 3, 0, [[LD2BASE]]
; CHECK: .LBB[[LAB1]]
; CHECK: xsadddp 0, 1, 2
; CHECK: xsadddp 1, 0, 3
; CHECK: blr
}
