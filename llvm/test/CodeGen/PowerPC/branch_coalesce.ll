; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mcpu=pwr8 -mtriple=powerpc64-unknown-linux-gnu -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu -verify-machineinstrs -disable-ppc-branch-coalesce < %s | FileCheck --check-prefix=CHECK-NOCOALESCE %s
; RUN: llc -mcpu=pwr8 -mtriple=powerpc64-unknown-linux-gnu -verify-machineinstrs -disable-ppc-branch-coalesce < %s | FileCheck --check-prefix=CHECK-NOCOALESCE %s

; Function Attrs: nounwind
define double @testBranchCoal(double %a, double %b, double %c, i32 %x) {

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

; CHECK-NOCOALESCE-LABEL: testBranchCoal:
; CHECK-NOCOALESCE:       # BB#0: # %entry
; CHECK-NOCOALESCE-NEXT:    cmplwi 0, 6, 0
; CHECK-NOCOALESCE-NEXT:    bne 0, .LBB0_5
; CHECK-NOCOALESCE-NEXT:  # BB#1: # %entry
; CHECK-NOCOALESCE-NEXT:    bne 0, .LBB0_6
; CHECK-NOCOALESCE-NEXT:  .LBB0_2: # %entry
; CHECK-NOCOALESCE-NEXT:    beq 0, .LBB0_4
; CHECK-NOCOALESCE-NEXT:  .LBB0_3: # %entry
; CHECK-NOCOALESCE-NEXT:    addis 3, 2, .LCPI0_1@toc@ha
; CHECK-NOCOALESCE-NEXT:    addi 3, 3, .LCPI0_1@toc@l
; CHECK-NOCOALESCE-NEXT:    lxsdx 3, 0, 3
; CHECK-NOCOALESCE-NEXT:  .LBB0_4: # %entry
; CHECK-NOCOALESCE-NEXT:    xsadddp 0, 1, 2
; CHECK-NOCOALESCE-NEXT:    xsadddp 1, 0, 3
; CHECK-NOCOALESCE-NEXT:    blr
; CHECK-NOCOALESCE-NEXT:  .LBB0_5: # %entry
; CHECK-NOCOALESCE-NEXT:    addis 3, 2, .LCPI0_0@toc@ha
; CHECK-NOCOALESCE-NEXT:    addi 3, 3, .LCPI0_0@toc@l
; CHECK-NOCOALESCE-NEXT:    lxsdx 1, 0, 3
; CHECK-NOCOALESCE-NEXT:    beq 0, .LBB0_2
; CHECK-NOCOALESCE-NEXT:  .LBB0_6: # %entry
; CHECK-NOCOALESCE-NEXT:    xxlxor 2, 2, 2
; CHECK-NOCOALESCE-NEXT:    bne 0, .LBB0_3
; CHECK-NOCOALESCE-NEXT:    b .LBB0_4
  entry:

  %test = icmp eq i32 %x, 0
  %tmp1 = select i1 %test, double %a, double 2.000000e-03
  %tmp2 = select i1 %test, double %b, double 0.000000e+00
  %tmp3 = select i1 %test, double %c, double 5.000000e-03

  %res1 = fadd double %tmp1, %tmp2
  %result = fadd double %res1, %tmp3
  ret double %result
}
