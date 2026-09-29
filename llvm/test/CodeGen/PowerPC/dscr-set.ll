; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu -mcpu=pwr9 \
; RUN: 	   | FileCheck %s -check-prefix=DEFAULT
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu -mcpu=pwr9 \
; RUN: 	   -ppc-set-dscr=0xFFFFFFFFFFFFFFFF | FileCheck %s -check-prefix=UPPER
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu -mcpu=pwr9 \
; RUN: 	   -ppc-set-dscr=0x01C4 | FileCheck %s -check-prefix=LOWER

define i32 @main() {

; CHECK:   	   # %bb.0:

; DEFAULT-NOT:	   mtudscr

; UPPER:	   lis [[inReg:[0-9]+]], 511
; UPPER-NEXT:	   ori [[inReg]], [[inReg]], 65535
; UPPER-NEXT:	   mtudscr [[inReg]]

; LOWER:	   lis [[inReg:[0-9]+]], 0
; LOWER-NEXT:	   ori [[inReg]], [[inReg]], 452
; LOWER-NEXT:	   mtudscr [[inReg]]

       ret i32 1
}
