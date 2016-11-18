; RUN: llc -march=bpfel < %s | FileCheck %s

define i16 @sccweqand(i16 %a, i16 %b) nounwind {
  %t1 = and i16 %a, %b
  %t2 = icmp eq i16 %t1, 0
  %t3 = zext i1 %t2 to i16
  ret i16 %t3
}
; CHECK-LABEL: sccweqand:
; CHECK: if r1 == r2

define i16 @sccwneand(i16 %a, i16 %b) nounwind {
  %t1 = and i16 %a, %b
  %t2 = icmp ne i16 %t1, 0
  %t3 = zext i1 %t2 to i16
  ret i16 %t3
}
; CHECK-LABEL: sccwneand:
; CHECK: if r1 != r2

define i16 @sccwne(i16 %a, i16 %b) nounwind {
  %t1 = icmp ne i16 %a, %b
  %t2 = zext i1 %t1 to i16
  ret i16 %t2
}
; CHECK-LABEL:sccwne:
; CHECK: if r1 != r2

define i16 @sccweq(i16 %a, i16 %b) nounwind {
  %t1 = icmp eq i16 %a, %b
  %t2 = zext i1 %t1 to i16
  ret i16 %t2
}
; CHECK-LABEL:sccweq:
; CHECK: if r1 == r2

define i16 @sccwugt(i16 %a, i16 %b) nounwind {
  %t1 = icmp ugt i16 %a, %b
  %t2 = zext i1 %t1 to i16
  ret i16 %t2
}
; CHECK-LABEL:sccwugt:
; CHECK: if r1 > r2

define i16 @sccwuge(i16 %a, i16 %b) nounwind {
  %t1 = icmp uge i16 %a, %b
  %t2 = zext i1 %t1 to i16
  ret i16 %t2
}
; CHECK-LABEL:sccwuge:
; CHECK: if r1 >= r2

define i16 @sccwult(i16 %a, i16 %b) nounwind {
  %t1 = icmp ult i16 %a, %b
  %t2 = zext i1 %t1 to i16
  ret i16 %t2
}
; CHECK-LABEL:sccwult:
; CHECK: if r2 > r1

define i16 @sccwule(i16 %a, i16 %b) nounwind {
  %t1 = icmp ule i16 %a, %b
  %t2 = zext i1 %t1 to i16
  ret i16 %t2
}
; CHECK-LABEL:sccwule:
; CHECK: if r2 >= r1

define i16 @sccwsgt(i16 %a, i16 %b) nounwind {
  %t1 = icmp sgt i16 %a, %b
  %t2 = zext i1 %t1 to i16
  ret i16 %t2
}
; CHECK-LABEL:sccwsgt:
; CHECK: if r1 s> r2

define i16 @sccwsge(i16 %a, i16 %b) nounwind {
  %t1 = icmp sge i16 %a, %b
  %t2 = zext i1 %t1 to i16
  ret i16 %t2
}
; CHECK-LABEL:sccwsge:
; CHECK: if r1 s>= r2

define i16 @sccwslt(i16 %a, i16 %b) nounwind {
  %t1 = icmp slt i16 %a, %b
  %t2 = zext i1 %t1 to i16
  ret i16 %t2
}
; CHECK-LABEL:sccwslt:
; CHECK: if r2 s> r1

define i16 @sccwsle(i16 %a, i16 %b) nounwind {
  %t1 = icmp sle i16 %a, %b
  %t2 = zext i1 %t1 to i16
  ret i16 %t2
}
; CHECK-LABEL:sccwsle:
; CHECK: if r2 s>= r1
