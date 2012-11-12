; RUN: llc < %s -march=x86 | FileCheck %s
; <rdar://problem/8285015>

define i32 @x(i32 %t) nounwind readnone ssp {
entry:
; CHECK: shll	$23, %eax
; CHECK: sarl	$31, %eax
; CHECK: andl	$-26, %eax
  %and = and i32 %t, 256
  %tobool = icmp eq i32 %and, 0
  %retval.0 = select i1 %tobool, i32 0, i32 -26
  ret i32 %retval.0
}
