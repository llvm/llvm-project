; RUN: opt < %s -passes=instcombine -S | FileCheck %s

define i32 @foo(i32 %x, i32 %y) {
; CHECK-LABEL: foo
; CHECK:      [[TMP1:%.*]] = icmp ugt i32 %x, %y
; CHECK-NEXT: [[TMP2:%.*]] = select i1 [[TMP1]], i32 %x, i32 %y, !prof ![[$MD0:[0-9]+]]
; CHECK-NEXT: [[TMP3:%.*]] = xor i32 [[TMP2]], -1
; CHECK-NEXT: ret i32 [[TMP3:%.*]]
; CHECK-DAG:  !0 = !{!"branch_weights", i32 6, i32 1}

  %1 = xor i32 %x, -1
  %2 = xor i32 %y, -1
  %3 = icmp ugt i32 %1, %2
  %4 = select i1 %3, i32 %2, i32 %1, !prof !1
  ret i32 %4
}

!1 = !{!"branch_weights", i32 1, i32 6}
