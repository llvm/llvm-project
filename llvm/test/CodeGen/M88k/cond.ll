; Test conditionals.
;
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88100 -verify-machineinstrs -m88k-enable-delay-slot-filler=false | FileCheck %s
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88110 -verify-machineinstrs -m88k-enable-delay-slot-filler=false | FileCheck %s

define i32 @f1(i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: bcnd ne0, %r2, [[LABEL:.LBB[_0-9]+]]
; CHECK: and %r2, %r2, %r3
; CHECK: jmp %r1
; CHECK: [[LABEL]]:
; CHECK: or %r2, %r0, 0
; CHECK: jmp %r1
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %compute, label %exit

compute:
  %res = and i32 %a, %b
  ret i32 %res

exit:
  ret i32 0
}

define i32 @f2(i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: cmp [[REG:%r[0-9]+]], %r2, %r3
; CHECK-NEXT: bb1 2, [[REG]], [[LABEL:.LBB[_0-9]+]]
; CHECK: and %r2, %r2, %r3
; CHECK: jmp %r1
; CHECK: [[LABEL]]:
; CHECK: or %r2, %r0, 0
; CHECK: jmp %r1
  %cmp = icmp ne i32 %a, %b
  br i1 %cmp, label %compute, label %exit

compute:
  %res = and i32 %a, %b
  ret i32 %res

exit:
  ret i32 0
}

define i32 @f3(i1 noundef zeroext %cnd) {
; CHECK-LABEL: f3:
; CHECK: bb0 0, %r2, [[LABEL:.LBB[_0-9]+]]
; CHECK: or %r2, %r0, 42
; CHECK: jmp %r1
; CHECK: [[LABEL]]:
; CHECK: or %r2, %r0, 84
; CHECK: jmp %r1
  br i1 %cnd, label %true, label %false

true:
  ret i32 42

false:
  ret i32 84
}

define i32 @f4(i1 noundef zeroext %cnd) {
; CHECK-LABEL: f4:
; CHECK: bb1 0, %r2, [[LABEL:.LBB[_0-9]+]]
; CHECK: or %r2, %r0, 42
; CHECK: jmp %r1
; CHECK: [[LABEL]]:
; CHECK: or %r2, %r0, 84
; CHECK: jmp %r1
  %not = xor i1 %cnd, true
  br i1 %not, label %true, label %false

true:
  ret i32 42

false:
  ret i32 84
}
