; RUN: llc -mtriple=bpfel -mcpu=v1 < %s | FileCheck --check-prefix=CHECK-V1 %s
; RUN: llc -mtriple=bpfel -mcpu=v2 < %s | FileCheck --check-prefix=CHECK-V2 %s

define i16 @sccweqand(i16 %a, i16 %b) nounwind {
  %t1 = and i16 %a, %b
  %t2 = icmp eq i16 %t1, 0
  %t3 = zext i1 %t2 to i16
  ret i16 %t3
}
; CHECK-LABEL: sccweqand:
; CHECK-V1: if r1 == 0
; CHECK-V2: if r1 == 0

define i16 @sccwneand(i16 %a, i16 %b) nounwind {
  %t1 = and i16 %a, %b
  %t2 = icmp ne i16 %t1, 0
  %t3 = zext i1 %t2 to i16
  ret i16 %t3
}
; CHECK-LABEL: sccwneand:
; CHECK-V1: if r1 != 0
; CHECK-V2: if r1 != 0

define i16 @sccwne(i16 %a, i16 %b) nounwind {
  %t1 = icmp ne i16 %a, %b
  %t2 = zext i1 %t1 to i16
  ret i16 %t2
}
; CHECK-LABEL:sccwne:
; CHECK-V1: if r1 != r2
; CHECK-V2: if r1 != r2

define i16 @sccweq(i16 %a, i16 %b) nounwind {
  %t1 = icmp eq i16 %a, %b
  %t2 = zext i1 %t1 to i16
  ret i16 %t2
}
; CHECK-LABEL:sccweq:
; CHECK-V1: if r1 == r2
; CHECK-V2: if r1 == r2

define i16 @sccwugt(i16 %a, i16 %b) nounwind {
  %t1 = icmp ugt i16 %a, %b
  %t2 = zext i1 %t1 to i16
  ret i16 %t2
}
; CHECK-LABEL:sccwugt:
; CHECK-V1: if r1 > r2
; CHECK-V2: if r1 > r2

define i16 @sccwuge(i16 %a, i16 %b) nounwind {
  %t1 = icmp uge i16 %a, %b
  %t2 = zext i1 %t1 to i16
  ret i16 %t2
}
; CHECK-LABEL:sccwuge:
; CHECK-V1: if r1 >= r2
; CHECK-V2: if r1 >= r2

define i16 @sccwult(i16 %a, i16 %b) nounwind {
  %t1 = icmp ult i16 %a, %b
  %t2 = zext i1 %t1 to i16
  ret i16 %t2
}
; CHECK-LABEL:sccwult:
; CHECK-V1: if r2 > r1
; CHECK-V2: if r1 < r2

define i16 @sccwule(i16 %a, i16 %b) nounwind {
  %t1 = icmp ule i16 %a, %b
  %t2 = zext i1 %t1 to i16
  ret i16 %t2
}
; CHECK-LABEL:sccwule:
; CHECK-V1: if r2 >= r1
; CHECK-V2: if r1 <= r2

define i16 @sccwsgt(i16 %a, i16 %b) nounwind {
  %t1 = icmp sgt i16 %a, %b
  %t2 = zext i1 %t1 to i16
  ret i16 %t2
}
; CHECK-LABEL:sccwsgt:
; CHECK-V1: if r1 s> r2
; CHECK-V2: if r1 s> r2

define i16 @sccwsge(i16 %a, i16 %b) nounwind {
  %t1 = icmp sge i16 %a, %b
  %t2 = zext i1 %t1 to i16
  ret i16 %t2
}
; CHECK-LABEL:sccwsge:
; CHECK-V1: if r1 s>= r2
; CHECK-V2: if r1 s>= r2

define i16 @sccwslt(i16 %a, i16 %b) nounwind {
  %t1 = icmp slt i16 %a, %b
  %t2 = zext i1 %t1 to i16
  ret i16 %t2
}
; CHECK-LABEL:sccwslt:
; CHECK-V1: if r2 s> r1
; CHECK-V2: if r1 s< r2

define i16 @sccwsle(i16 %a, i16 %b) nounwind {
  %t1 = icmp sle i16 %a, %b
  %t2 = zext i1 %t1 to i16
  ret i16 %t2
}
; CHECK-LABEL:sccwsle:
; CHECK-V1: if r2 s>= r1
; CHECK-V2: if r1 s<= r2
