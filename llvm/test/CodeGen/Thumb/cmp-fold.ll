; RUN: llc -mtriple=thumbv6m-eabi -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: subs:
; CHECK: subs
; CHECK-NEXT: b{{eq|ne}}
define i32 @subs(i32 %a, i32 %b) {
  %c = sub i32 %a, %b
  %d = icmp eq i32 %c, 0
  br i1 %d, label %true, label %false

true:
  ret i32 4
false:
  ret i32 5
}

; CHECK-LABEL: addsrr:
; CHECK: adds
; CHECK-NEXT: b{{eq|ne}}
define i32 @addsrr(i32 %a, i32 %b) {
  %c = add i32 %a, %b
  %d = icmp eq i32 %c, 0
  br i1 %d, label %true, label %false

true:
  ret i32 4
false:
  ret i32 5
}

; CHECK-LABEL: lslri:
; CHECK: lsls
; CHECK-NEXT: b{{eq|ne}}
define i32 @lslri(i32 %a, i32 %b) {
  %c = shl i32 %a, 3
  %d = icmp eq i32 %c, 0
  br i1 %d, label %true, label %false

true:
  ret i32 4
false:
  ret i32 5
}

; CHECK-LABEL: lslrr:
; CHECK: lsls
; CHECK-NEXT: b{{eq|ne}}
define i32 @lslrr(i32 %a, i32 %b) {
  %c = shl i32 %a, %b
  %d = icmp eq i32 %c, 0
  br i1 %d, label %true, label %false

true:
  ret i32 4
false:
  ret i32 5
}
