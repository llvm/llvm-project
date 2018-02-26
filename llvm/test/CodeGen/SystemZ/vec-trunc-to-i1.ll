; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s
;
; Check that a widening truncate to a vector of i1 elements can be handled.

; NOTE: REG2 is actually not needed (tempororary FAIL)
define void @pr32275(<4 x i8> %B15) {
; CHECK-LABEL: pr32275:
; CHECK:       # %bb.0: # %BB
; CHECK-NEXT:    vrepif [[REG0:%v[0-9]]], 1
; CHECK:         vlgvb %r0, %v24, 3
; CHECK-NEXT:    vlgvb %r1, %v24, 1
; CHECK-NEXT:    vlvgp [[REG1:%v[0-9]]], %r1, %r0
; CHECK-NEXT:    vlgvb %r0, %v24, 0
; CHECK-DAG:     vlr [[REG2:%v[0-9]]], [[REG1]]
; CHECK-DAG:     vlvgf [[REG2]], %r0, 0
; CHECK-DAG:     vlgvb [[REG3:%r[0-9]]], %v24, 2
; CHECK-NEXT:    vlvgf [[REG2]], [[REG3]], 2
; CHECK-NEXT:    vn [[REG2]], [[REG2]], [[REG0]]
; CHECK-NEXT:    vlgvf [[REG4:%r[0-9]]], [[REG2]], 3
; CHECK-NEXT:    tmll [[REG4]], 1
; CHECK-NEXT:    jne .LBB0_1
; CHECK-NEXT:  # %bb.2: # %CF36
; CHECK-NEXT:    br %r14
BB:
  br label %CF34

CF34:
  %Tr24 = trunc <4 x i8> %B15 to <4 x i1>
  %E28 = extractelement <4 x i1> %Tr24, i32 3
  br i1 %E28, label %CF34, label %CF36

CF36:
  ret void
}
