; RUN: llc < %s -mtriple=i686-- | FileCheck %s

; Don't duplicate the load.

define fastcc i32 @foo(i32* %p) nounwind {
; CHECK-LABEL: foo:
; CHECK: andl $10, %eax
; CHECK: je
	%t0 = load i32, i32* %p
	%t2 = and i32 %t0, 10
	%t3 = icmp ne i32 %t2, 0
	br i1 %t3, label %bb63, label %bb76
bb63:
	ret i32 %t2
bb76:
	ret i32 0
}

define fastcc double @bar(i32 %hash, double %x, double %y) nounwind {
entry:
; CHECK-LABEL: bar:
  %0 = and i32 %hash, 15
  %1 = icmp ult i32 %0, 8
  br i1 %1, label %bb11, label %bb10

bb10:
; CHECK: bb10
; CHECK: testb $1
  %2 = and i32 %hash, 1
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %bb13, label %bb11

bb11:
  %4 = fsub double -0.000000e+00, %x
  br label %bb13

bb13:
; CHECK: bb13
; CHECK: testb $2
  %iftmp.9.0 = phi double [ %4, %bb11 ], [ %x, %bb10 ]
  %5 = and i32 %hash, 2
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %bb16, label %bb14

bb14:
  %7 = fsub double -0.000000e+00, %y
  br label %bb16

bb16:
  %iftmp.10.0 = phi double [ %7, %bb14 ], [ %y, %bb13 ]
  %8 = fadd double %iftmp.9.0, %iftmp.10.0
  ret double %8
}
