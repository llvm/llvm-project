; RUN: llc < %s -mtriple=x86_64-apple-macosx | FileCheck %s

; rdar://12081007

; CHECK: and_1:
; CHECK: andb
; CHECK-NEXT: cmovnel
; CHECK: ret
define i32 @and_1(i8 zeroext %a, i8 zeroext %b, i32 %x) {
  %1 = and i8 %b, %a
  %2 = icmp ne i8 %1, 0
  %3 = select i1 %2, i32 %x, i32 0
  ret i32 %3
}

; CHECK: and_2:
; CHECK: andb
; CHECK-NEXT: setne
; CHECK: ret
define zeroext i1 @and_2(i8 zeroext %a, i8 zeroext %b) {
  %1 = and i8 %b, %a
  %2 = icmp ne i8 %1, 0
  ret i1 %2
}

; CHECK: xor_1:
; CHECK: xorb
; CHECK-NEXT: cmovnel
; CHECK: ret
define i32 @xor_1(i8 zeroext %a, i8 zeroext %b, i32 %x) {
  %1 = xor i8 %b, %a
  %2 = icmp ne i8 %1, 0
  %3 = select i1 %2, i32 %x, i32 0
  ret i32 %3
}

; CHECK: xor_2:
; CHECK: xorb
; CHECK-NEXT: setne
; CHECK: ret
define zeroext i1 @xor_2(i8 zeroext %a, i8 zeroext %b) {
  %1 = xor i8 %b, %a
  %2 = icmp ne i8 %1, 0
  ret i1 %2
}
