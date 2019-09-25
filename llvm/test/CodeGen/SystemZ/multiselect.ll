; Test that multiple select statements using the same condition are expanded
; into a single conditional branch when possible.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -disable-block-placement | FileCheck %s

define void @test0(i32 signext %positive, double %base, double %offset, double* %rmin, double* %rmax) {
entry:
; CHECK-LABEL: test0
; CHECK: cijlh %r2, 0,
; CHECK-NOT: cij
; CHECK-NOT: je
; CHECK-NOT: jlh

  %tobool = icmp eq i32 %positive, 0
  %add = fadd double %base, %offset
  %min = select i1 %tobool, double %add, double %base
  %max = select i1 %tobool, double %base, double %add
  store double %min, double* %rmin, align 8
  store double %max, double* %rmax, align 8
  ret void
}

; Two selects with an intervening instruction that doesn't clobber CC can
; still be merged.
define double @test1(i32 signext %positive, double %A, double %B, double %C) {
entry:
; CHECK-LABEL: test1
; CHECK: cijhe {{.*}}LBB1_2
; CHECK-NOT: cij
; CHECK: br %r14

  %tobool = icmp slt i32 %positive, 0
  %s1  = select i1 %tobool, double %A, double %B
  %mul = fmul double %A, %B
  %s2  = select i1 %tobool, double %B, double %C
  %add = fadd double %s1, %s2
  %add2 = fadd double %add, %mul
  ret double %add2
}

; Two selects with an intervening user of the first select can't be merged.
define double @test2(i32 signext %positive, double %A, double %B) {
entry:
; CHECK-LABEL: test2
; CHECK: cije {{.*}}LBB2_2
; CHECK: cibe {{.*}}%r14
; CHECK: br %r14

  %tobool = icmp eq i32 %positive, 0
  %s1  = select i1 %tobool, double %A, double %B
  %add = fadd double %A, %s1
  %s2  = select i1 %tobool, double %A, double %add
  ret double %s2
}

; Two selects with different conditions can't be merged
define double @test3(i32 signext %positive, double %A, double %B, double %C) {
entry:
; CHECK-LABEL: test3
; CHECK: cijl {{.*}}LBB3_2
; CHECK: cijl {{.*}}LBB3_4
; CHECK: br %r14

  %tobool = icmp slt i32 %positive, 0
  %s1  = select i1 %tobool, double %A, double %B
  %tobool2 = icmp slt i32 %positive, 2
  %s2  = select i1 %tobool2, double %B, double %C
  %add = fadd double %s1, %s2
  ret double %add
}
