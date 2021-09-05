; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s
; RUN: llvm-mc -triple=m68k -filetype=obj < %s | \
; RUN:     llvm-objdump -d - | FileCheck --check-prefix=CHECK-OBJ %s

; CHECK-LABEL: BACKWARD:
BACKWARD:
	; CHECK:      nop
	; CHECK-SAME: encoding: [0x4e,0x71]
	nop
	; CHECK:      rts
	; CHECK-SAME: encoding: [0x4e,0x75]
	rts

; CHECK-LABEL: PCI:
PCI:
	; CHECK:     jsr  (BACKWARD,%pc,%d0)
        ; CHECK-OBJ: jsr  (250,%pc,%d0)
	; CHECK-SAME: encoding: [0x4e,0xbb,0x08,A]
        ; CHECK:      fixup A - offset: 3, value: BACKWARD+1, kind: FK_PCRel_1
	jsr	(BACKWARD,%pc,%d0)
	; CHECK:     jsr  (FORWARD,%pc,%d0)
        ; CHECK-OBJ: jsr  (10,%pc,%d0)
	; CHECK-SAME: encoding: [0x4e,0xbb,0x08,A]
        ; CHECK:      fixup A - offset: 3, value: FORWARD+1, kind: FK_PCRel_1
	jsr	(FORWARD,%pc,%d0)

; CHECK-LABEL: PCD:
PCD:
	; CHECK:     jsr  (BACKWARD,%pc)
	; CHECK-OBJ: jsr  (65522,%pc)
	; CHECK-SAME: encoding: [0x4e,0xba,A,A]
        ; CHECK:      fixup A - offset: 2, value: BACKWARD, kind: FK_PCRel_2
	jsr	(BACKWARD,%pc)
	; CHECK:     jsr  (FORWARD,%pc)
	; CHECK-OBJ: jsr  (2,%pc)
	; CHECK-SAME: encoding: [0x4e,0xba,A,A]
        ; CHECK:      fixup A - offset: 2, value: FORWARD, kind: FK_PCRel_2
	jsr	(FORWARD,%pc)

; CHECK-LABEL: FORWARD:
FORWARD:
	; CHECK:      nop
	; CHECK-SAME: encoding: [0x4e,0x71]
	nop
	; CHECK:      rts
	; CHECK-SAME: encoding: [0x4e,0x75]
	rts

