; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a8 %s -show-mc-encoding -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-LE
; RUN: llc -mtriple=armeb-eabi -mcpu=cortex-a8 %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-BE
; RUN: llc -mtriple=thumbv6m %s -o - | FileCheck %s --check-prefix=CHECK-V6M
; RUN: llc -mtriple=thumbv8m.base -show-mc-encoding %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V8M
; RUN: llc -mtriple=thumbv8m.main -show-mc-encoding %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V8M

; 171 = 0x000000ab
define i64 @f1(i64 %a) {
; CHECK-LABEL: f1
; CHECK-LE: subs{{.*}} r0, #171
; CHECK-LE: sbc r1, r1, #0
; CHECK-BE: subs r1, r1, #171
; CHECK-BE: sbc r0, r0, #0
    %tmp = sub i64 %a, 171
    ret i64 %tmp
}

; 66846720 = 0x03fc0000
define i64 @f2(i64 %a) {
; CHECK-LABEL: f2
; CHECK-LE: subs{{.*}} r0, r0, #66846720
; CHECK-LE: sbc r1, r1, #0
; CHECK-BE: subs r1, r1, #66846720
; CHECK-BE: sbc r0, r0, #0
    %tmp = sub i64 %a, 66846720
    ret i64 %tmp
}

; 734439407618 = 0x000000ab00000002
define i64 @f3(i64 %a) {
; CHECK-LABEL: f3
; CHECK-LE: subs{{.*}} r0, #2
; CHECK-LE: sbc r1, r1, #171
; CHECK-BE: subs r1, r1, #2
; CHECK-BE: sbc r0, r0, #171
   %tmp = sub i64 %a, 734439407618
   ret i64 %tmp
}

define i32 @f4(i32 %x) {
entry:
; CHECK-LABEL: f4
; CHECK-LE: rsbs
; CHECK-BE: rsbs
  %sub = sub i32 1, %x
  %cmp = icmp ugt i32 %sub, 0
  %sel = select i1 %cmp, i32 1, i32 %sub
  ret i32 %sel
}

define i32 @f5(i32 %x) {
entry:
; CHECK-LABEL: f5:
; CHECK-LE:  movw r1, #65535 @ encoding: [0xff,0x1f,0x0f,0xe3]
; CHECK-V8M: movw r1, #65535 @ encoding: [0x4f,0xf6,0xff,0x71]
; CHECK-NOT: movt
; CHECK-NOT: add
; CHECK: sub{{.*}} r0, r0, r1

; CHECK-V6M-LABEL: f5
; CHECK-V6M: ldr [[NEG:r[0-1]+]], [[CONST:.[A-Z0-9_]+]]
; CHECK-V6M: add{{.*}} r0, [[NEG]]
; CHECK-V6M: [[CONST]]
; CHECK-V6M: .long   4294901761
  %sub = add i32 %x, -65535
  ret i32 %sub
}

define i32 @f6(i32 %x) {
entry:
; CHECK-LABEL: f6:
; CHECK-LE:  movw r1, #65535 @ encoding: [0xff,0x1f,0x0f,0xe3]
; CHECK-V8M: movw r1, #65535 @ encoding: [0x4f,0xf6,0xff,0x71]
; CHECK-NOT: movt
; CHECK-NOT: sub
; CHECK: add{{.*}} r0, r1

; CHECK-V6M-LABEL: f6
; CHECK-V6M: ldr [[NEG:r[0-1]+]], [[CONST:.[A-Z0-9_]+]]
; CHECK-V6M: add{{.*}} r0, [[NEG]]
; CHECK-V6M: [[CONST]]
; CHECK-V6M: .long 65535
  %sub = sub i32 %x, -65535
  ret i32 %sub
}
