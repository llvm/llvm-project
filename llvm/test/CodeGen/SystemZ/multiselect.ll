; Test that multiple select statements using the same condition are expanded
; into a single conditional branch.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @test(i32 signext %positive, double %base, double %offset, double* %rmin, double* %rmax) {
entry:
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

