; RUN: llc -march=arm -mcpu=cortex-a8 < %s | FileCheck %s

; 171 = 0x000000ab
define i64 @f1(i64 %a) {
; CHECK: f1
; CHECK: subs r0, r0, #171
; CHECK: sbc r1, r1, #0
    %tmp = sub i64 %a, 171
    ret i64 %tmp
}

; 66846720 = 0x03fc0000
define i64 @f2(i64 %a) {
; CHECK: f2
; CHECK: subs r0, r0, #66846720
; CHECK: sbc r1, r1, #0
    %tmp = sub i64 %a, 66846720
    ret i64 %tmp
}

; 734439407618 = 0x000000ab00000002
define i64 @f3(i64 %a) {
; CHECK: f3
; CHECK: subs r0, r0, #2
; CHECK: sbc r1, r1, #171
   %tmp = sub i64 %a, 734439407618
   ret i64 %tmp
}

define i32 @f4(i32 %x) {
entry:
; CHECK: f4
; CHECK: rsbs
  %sub = sub i32 1, %x
  %cmp = icmp ugt i32 %sub, 0
  %sel = select i1 %cmp, i32 1, i32 %sub
  ret i32 %sel
}

; rdar://11726136
define i32 @f5(i32 %x) {
entry:
; CHECK: f5
; CHECK: movw r1, #65535
; CHECK-NOT: movt
; CHECK-NOT: add
; CHECK: sub r0, r0, r1
  %sub = add i32 %x, -65535
  ret i32 %sub
}
