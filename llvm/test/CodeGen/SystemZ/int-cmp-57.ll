; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 -disable-cgp | FileCheck %s
;
; Check that signed comparisons against 0 are eliminated if the defining
; instruction is an add with immediate.
;
; Addition of an immediate does not depend on the "nsw" flag, since the
; result can be predicted in case of overflow. For example, if adding a
; positive immediate gives overflow, the result must be negative.

; Addition of a negative immediate gives a positive result in case of
; overflow (except for the case of the minimum value which may also result in
; a zero result).
define i32 @fun0(i32 %arg) {
; CHECK-LABEL: fun0:
; CHECK: ahik
; CHECK-NEXT: locre
bb:
  %tmp = add i32 %arg, -1
  %tmp1 = icmp eq i32 %tmp, 0
  %res = select i1 %tmp1, i32 %tmp, i32 %arg
  ret i32 %res
}

define i32 @fun1(i32 %arg) {
; CHECK-LABEL: fun1:
; CHECK: ahik
; CHECK-NEXT: locrnle
bb:
  %tmp = add i32 %arg, -1
  %tmp1 = icmp sgt i32 %tmp, 0
  %res = select i1 %tmp1, i32 %tmp, i32 %arg
  ret i32 %res
}

define i32 @fun2(i32 %arg) {
; CHECK-LABEL: fun2:
; CHECK: ahik
; CHECK-NEXT: locrl
bb:
  %tmp = add i32 %arg, -1
  %tmp1 = icmp slt i32 %tmp, 0
  %res = select i1 %tmp1, i32 %tmp, i32 %arg
  ret i32 %res
}

; Addition of a positive immediate gives a negative result in case of overflow.
define i32 @fun3(i32 %arg) {
; CHECK-LABEL: fun3:
; CHECK: ahik
; CHECK-NEXT: locre
bb:
  %tmp = add i32 %arg, 1
  %tmp1 = icmp eq i32 %tmp, 0
  %res = select i1 %tmp1, i32 %tmp, i32 %arg
  ret i32 %res
}

define i32 @fun4(i32 %arg) {
; CHECK-LABEL: fun4:
; CHECK: ahik
; CHECK-NEXT: locrh
bb:
  %tmp = add i32 %arg, 1
  %tmp1 = icmp sgt i32 %tmp, 0
  %res = select i1 %tmp1, i32 %tmp, i32 %arg
  ret i32 %res
}

define i32 @fun5(i32 %arg) {
; CHECK-LABEL: fun5:
; CHECK: ahik
; CHECK-NEXT: locrnhe
bb:
  %tmp = add i32 %arg, 1
  %tmp1 = icmp slt i32 %tmp, 0
  %res = select i1 %tmp1, i32 %tmp, i32 %arg
  ret i32 %res
}

; Addition of the minimum value gives a positive or zero result.
define i32 @fun6(i32 %arg) {
; CHECK-LABEL: fun6:
; CHECK: afi
; CHECK-NEXT: chi
; CHECK-NEXT: locre
bb:
  %tmp = add i32 %arg, -2147483648
  %tmp1 = icmp eq i32 %tmp, 0
  %res = select i1 %tmp1, i32 %tmp, i32 %arg
  ret i32 %res
}

define i32 @fun7(i32 %arg) {
; CHECK-LABEL: fun7:
; CHECK: afi
; CHECK-NEXT: chi
; CHECK-NEXT: locrh
bb:
  %tmp = add i32 %arg, -2147483648
  %tmp1 = icmp sgt i32 %tmp, 0
  %res = select i1 %tmp1, i32 %tmp, i32 %arg
  ret i32 %res
}
