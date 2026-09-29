; Test insertions of 16-bit constants into an i64.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check the lowest useful IILL value.  (We use NILL rather than IILL
; to clear 16 bits.)
define i64 @f1(i64 %a) {
; CHECK-LABEL: f1:
; CHECK-NOT: ni
; CHECK: iill %r2, 1
; CHECK: br %r14
  %and = and i64 %a, 18446744073709486080
  %or = or i64 %and, 1
  ret i64 %or
}

; Check a middle value.
define i64 @f2(i64 %a) {
; CHECK-LABEL: f2:
; CHECK-NOT: ni
; CHECK: iill %r2, 32769
; CHECK: br %r14
  %and = and i64 %a, -65536
  %or = or i64 %and, 32769
  ret i64 %or
}

; Check the highest useful IILL value.  (We use OILL rather than IILL
; to set 16 bits.)
define i64 @f3(i64 %a) {
; CHECK-LABEL: f3:
; CHECK-NOT: ni
; CHECK: iill %r2, 65534
; CHECK: br %r14
  %and = and i64 %a, 18446744073709486080
  %or = or i64 %and, 65534
  ret i64 %or
}

; Check the lowest useful IILH value.
define i64 @f4(i64 %a) {
; CHECK-LABEL: f4:
; CHECK-NOT: ni
; CHECK: iilh %r2, 1
; CHECK: br %r14
  %and = and i64 %a, 18446744069414649855
  %or = or i64 %and, 65536
  ret i64 %or
}

; Check a middle value.
define i64 @f5(i64 %a) {
; CHECK-LABEL: f5:
; CHECK-NOT: ni
; CHECK: iilh %r2, 32767
; CHECK: br %r14
  %and = and i64 %a, -4294901761
  %or = or i64 %and, 2147418112
  ret i64 %or
}

; Check the highest useful IILH value.
define i64 @f6(i64 %a) {
; CHECK-LABEL: f6:
; CHECK-NOT: ni
; CHECK: iilh %r2, 65534
; CHECK: br %r14
  %and = and i64 %a, 18446744069414649855
  %or = or i64 %and, 4294836224
  ret i64 %or
}

; Check the lowest useful IIHL value.
define i64 @f7(i64 %a) {
; CHECK-LABEL: f7:
; CHECK-NOT: ni
; CHECK: iihl %r2, 1
; CHECK: br %r14
  %and = and i64 %a, 18446462603027808255
  %or = or i64 %and, 4294967296
  ret i64 %or
}

; Check a middle value.
define i64 @f8(i64 %a) {
; CHECK-LABEL: f8:
; CHECK-NOT: ni
; CHECK: iihl %r2, 32767
; CHECK: br %r14
  %and = and i64 %a, -281470681743361
  %or = or i64 %and, 140733193388032
  ret i64 %or
}

; Check the highest useful IIHL value.
define i64 @f9(i64 %a) {
; CHECK-LABEL: f9:
; CHECK-NOT: ni
; CHECK: iihl %r2, 65534
; CHECK: br %r14
  %and = and i64 %a, 18446462603027808255
  %or = or i64 %and, 281466386776064
  ret i64 %or
}

; Check the lowest useful IIHH value.
define i64 @f10(i64 %a) {
; CHECK-LABEL: f10:
; CHECK-NOT: ni
; CHECK: iihh %r2, 1
; CHECK: br %r14
  %and = and i64 %a, 281474976710655
  %or = or i64 %and, 281474976710656
  ret i64 %or
}

; Check a middle value.
define i64 @f11(i64 %a) {
; CHECK-LABEL: f11:
; CHECK-NOT: ni
; CHECK: iihh %r2, 32767
; CHECK: br %r14
  %and = and i64 %a, 281474976710655
  %or = or i64 %and, 9223090561878065152
  ret i64 %or
}

; Check the highest useful IIHH value.
define i64 @f12(i64 %a) {
; CHECK-LABEL: f12:
; CHECK-NOT: ni
; CHECK: iihh %r2, 65534
; CHECK: br %r14
  %and = and i64 %a, 281474976710655
  %or = or i64 %and, 18446181123756130304
  ret i64 %or
}
