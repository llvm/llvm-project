; RUN: llc < %s -mtriple=thumbv7 -o - | llvm-mc -triple thumbv7 --show-encoding 2>&1 | FileCheck %s --check-prefix=V7
; RUN: llc < %s -mtriple=thumbv7 -arm-restrict-it -o - | llvm-mc -triple thumbv7 --show-encoding 2>&1 | FileCheck %s --check-prefix=V7_RESTRICT_IT
; RUN: llc < %s -mtriple=thumbv8 -o - | llvm-mc -triple thumbv8 --show-encoding 2>&1 | FileCheck %s --check-prefix=V8
; RUN: llc < %s -mtriple=thumbv8 -arm-no-restrict-it -o - | llvm-mc -triple thumbv8 --show-encoding 2>&1 | FileCheck %s --check-prefix=V8_NO_RESTRICT_IT


; V7-NOT: warning
; V7_RESTRICT_IT-NOT: warning
; V8-NOT: warning
; V8_NO_RESTRICT_IT: warning: deprecated instruction in IT block
; it ge                           @ encoding: [0xa8,0xbf]
; lslge.w r3, r12, lr             @ encoding: [0x0c,0xfa,0x0e,0xf3]   ; deprecated in ARMv8 thumb mode
define i1 @scalar_i64_lowestbit_eq(i64 %x, i64 %y) {
%t0 = shl i64 1, %y
%t1 = and i64 %t0, %x
%res = icmp eq i64 %t1, 0
ret i1 %res
}

; V7-NOT: warning
; V7_RESTRICT_IT-NOT: warning
; V8-NOT: warning
; V8_NO_RESTRICT_IT: warning: deprecated instruction in IT block
; it ne                           @ encoding: [0x18,0xbf]
; movne.w r0, #-1                 @ encoding: [0x4f,0xf0,0xff,0x30]   ; deprecated in ARMv8 thumb mode
define i32 @icmp_eq_minus_one(i8* %ptr) {
  %load = load i8, i8* %ptr, align 1
  %conv = zext i8 %load to i32
  %cmp = icmp eq i8 %load, -1
  %ret = select i1 %cmp, i32 %conv, i32 -1
  ret i32 %ret
}
