; RUN: llc -mtriple=thumbv6m-eabi -verify-machineinstrs < %s | FileCheck --check-prefix=CHECK --check-prefix=T1 %s
; RUN: llc -mtriple=thumbv7m-eabi -verify-machineinstrs < %s | FileCheck --check-prefix=CHECK --check-prefix=T2 %s

; CHECK-LABEL: addri1:
; CHECK: adds r0, #3
; T1-NEXT: b{{eq|ne}}
; T2-NOT: cmp
define i32 @addri1(i32 %a, i32 %b) {
  %c = add i32 %a, 3
  %d = icmp eq i32 %c, 0
  br i1 %d, label %true, label %false

true:
  ret i32 4
false:
  ret i32 5
}

; CHECK-LABEL: addri2:
; CHECK: adds r0, #254
; T1-NEXT: b{{eq|ne}}
; T2-NOT: cmp
define i32 @addri2(i32 %a, i32 %b) {
  %c = add i32 %a, 254
  %d = icmp eq i32 %c, 0
  br i1 %d, label %true, label %false

true:
  ret i32 4
false:
  ret i32 5
}
