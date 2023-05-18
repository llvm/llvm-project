; Test insertions of 16-bit constants into one half of an i32.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check the lowest useful IILL value.  (We use NILL rather than IILL
; to clear 16 bits.)
define i32 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK-NOT: ni
; CHECK: iill %r2, 1
; CHECK: br %r14
  %and = and i32 %a, 4294901760
  %or = or i32 %and, 1
  ret i32 %or
}

; Check a middle value.
define i32 @f2(i32 %a) {
; CHECK-LABEL: f2:
; CHECK-NOT: ni
; CHECK: iill %r2, 32769
; CHECK: br %r14
  %and = and i32 %a, -65536
  %or = or i32 %and, 32769
  ret i32 %or
}

; Check the highest useful IILL value.  (We use OILL rather than IILL
; to set 16 bits.)
define i32 @f3(i32 %a) {
; CHECK-LABEL: f3:
; CHECK-NOT: ni
; CHECK: iill %r2, 65534
; CHECK: br %r14
  %and = and i32 %a, 4294901760
  %or = or i32 %and, 65534
  ret i32 %or
}

; Check the lowest useful IILH value.
define i32 @f4(i32 %a) {
; CHECK-LABEL: f4:
; CHECK-NOT: ni
; CHECK: iilh %r2, 1
; CHECK: br %r14
  %and = and i32 %a, 65535
  %or = or i32 %and, 65536
  ret i32 %or
}

; Check a middle value.
define i32 @f5(i32 %a) {
; CHECK-LABEL: f5:
; CHECK-NOT: ni
; CHECK: iilh %r2, 32767
; CHECK: br %r14
  %and = and i32 %a, 65535
  %or = or i32 %and, 2147418112
  ret i32 %or
}

; Check the highest useful IILH value.
define i32 @f6(i32 %a) {
; CHECK-LABEL: f6:
; CHECK-NOT: ni
; CHECK: iilh %r2, 65534
; CHECK: br %r14
  %and = and i32 %a, 65535
  %or = or i32 %and, -131072
  ret i32 %or
}
