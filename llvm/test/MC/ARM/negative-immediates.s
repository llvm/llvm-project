# RUN: llvm-mc -triple armv7 %s -show-encoding | FileCheck %s
# RUN: not llvm-mc -triple armv7 %s -show-encoding -mattr=+no-neg-immediates 2>&1 | FileCheck %s -check-prefix=CHECK-DISABLED

.arm

	ADC r0, r1, #0xFFFFFF00
# CHECK: sbc r0, r1, #255
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: ADC
	ADC r0, r1, #0xFFFFFE03
# CHECK: sbc r0, r1, #508
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: ADC
	ADD r0, r1, #0xFFFFFF01
# CHECK: sub r0, r1, #255
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: ADD
	AND r0, r1, #0xFFFFFF00
# CHECK: bic r0, r1, #255
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: AND
	BIC r0, r1, #0xFFFFFF00
# CHECK: and r0, r1, #255
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: BIC
	CMP r0, #0xFFFFFF01
# CHECK: cmn r0, #255
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: CMP
	CMN r0, #0xFFFFFF01
# CHECK: cmp r0, #255
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: CMN
	MOV r0, #0xFFFFFF00
# CHECK: mvn r0, #255
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: MOV
	MVN r0, #0xFFFFFF00
# CHECK: mov r0, #255
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: MVN
	SBC r0, r1, #0xFFFFFF00
# CHECK: adc r0, r1, #255
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: SBC
	SUB r0, r1, #0xFFFFFF01
# CHECK: add r0, r1, #255
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: SUB

.thumb

	ADC r0, r1, #0xFFFFFF00
# CHECK: sbc r0, r1, #255
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: ADC
	ADC r0, r1, #0xFFFF00FF
# CHECK: sbc r0, r1, #65280
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: ADC
	ADC r0, r1, #0xFFFEFFFE
# CHECK: sbc r0, r1, #65537 @ encoding: [0x61,0xf1,0x01,0x10]
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: ADC
	ADC r0, r1, #0xFEFFFEFF
# CHECK: sbc r0, r1, #16777472 @ encoding: [0x61,0xf1,0x01,0x20]
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: ADC
	ADD.W r0, r0, #0xFFFFFF01
# CHECK: sub.w r0, r0, #255
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: ADD.W
	ADD.W r0, r0, #0xFF01FF02
# CHECK: sub.w r0, r0, #16646398 @ encoding: [0xa0,0xf1,0xfe,0x10]
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: ADD.W
	ADDW r0, r1, #0xFFFFFF01
# CHECK: subw r0, r1, #255 @ encoding: [0xa1,0xf2,0xff,0x00]
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: ADDW
	ADD.W r0, r1, #0xFFFFFF01
# CHECK: sub.w r0, r1, #255 @ encoding: [0xa1,0xf1,0xff,0x00]
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: ADD.W
	AND r0, r1, #0xFFFFFF00
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: AND
# CHECK: bic r0, r1, #255
	AND r0, r1, #0xFEFFFEFF
# CHECK: bic r0, r1, #16777472 @ encoding: [0x21,0xf0,0x01,0x20]
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: AND
	BIC r0, r1, #0xFFFFFF00
# CHECK: and r0, r1, #255
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: BIC
	BIC r0, r1, #0xFEFFFEFF
# CHECK: and r0, r1, #16777472 @ encoding: [0x01,0xf0,0x01,0x20]
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: BIC
	ORR r0, r1, #0xFFFFFF00
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: ORR
# CHECK: orn r0, r1, #255
	ORR r0, r1, #0xFEFFFEFF
# CHECK: orn r0, r1, #16777472 @ encoding: [0x61,0xf0,0x01,0x20]
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: ORR
	ORN r0, r1, #0xFFFFFF00
# CHECK: orr r0, r1, #255
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: ORN
	ORN r0, r1, #0xFEFFFEFF
# CHECK: orr r0, r1, #16777472 @ encoding: [0x41,0xf0,0x01,0x20]
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: ORN
	CMP r0, #0xFFFFFF01
# CHECK: cmn.w r0, #255
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: CMP
	CMN r0, #0xFFFFFF01
# CHECK: cmp.w r0, #255
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: CMN
	MOV r0, #0xFFFFFF00
# CHECK: mvn r0, #255
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: MOV
	MVN r0, #0xFFFFFF00
# CHECK: mov.w r0, #255
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: MVN
	SBC r0, r1, #0xFFFFFF00
# CHECK: adc r0, r1, #255
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: SBC
	SUBW r0, r1, #0xFFFFFF01
# CHECK: addw r0, r1, #255
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: SUBW
	SUB.W r0, r1, #0xFFFFFF01
# CHECK: add.w r0, r1, #255
# CHECK-DISABLED: note: instruction requires: NegativeImmediates
# CHECK-DISABLED: SUB.W
