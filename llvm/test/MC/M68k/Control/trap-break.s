; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      trap #13
; CHECK-SAME: encoding: [0x4e,0x4d]
trap	#13
; CHECK:      bkpt #7
; CHECK-SAME: encoding: [0x48,0x4f]
bkpt	#7
; CHECK:      trapv
; CHECK-SAME: encoding: [0x4e,0x76]
trapv
; CHECK:      illegal
; CHECK-SAME: encoding: [0x4a,0xfc]
illegal
