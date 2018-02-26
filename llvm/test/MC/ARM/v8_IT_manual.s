@ RUN: llvm-mc -triple thumbv8 -show-encoding < %s 2>&1 | FileCheck %s

@ ADD reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
addge r1, r2, r3
@ ADD reg, encoding T2
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
addge r1, r2
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge r1, pc
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge pc, r2
@ ADD reg, encoding T3 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge r11, r2, r3
@ ADD imm, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
addge r1, r2, #3
@ ADD imm, encoding T2
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
addge r1, #3
@ ADD imm, encoding T3 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge r11, r2, #3
@ ADD imm, encoding T4 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge r11, r2, #333
@ ADD SP+imm, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
addge r1, sp, #32
@ ADD SP+imm, encoding T2
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge sp, #32
@ ADD SP+imm, encoding T3 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge r1, sp, #33
@ ADD SP+imm, encoding T4 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge r1, sp, #333

@ SUB reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
subge r4, r3, r2
@ SUB reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge r14, r3, r2
@ SUB imm, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
subge r4, r3, #2
@ SUB imm, encoding T2
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
subge r4, #3
@ SUB imm, encoding T3 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge r14, r3, #2
@ SUB imm, encoding T4 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge r14, r3, #2222
@ SUB SP-imm, encoding T1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge sp, #32
@ SUB SP-imm, encoding T3 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge r4, sp, #33
@ SUB SP-imm, encoding T4 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge r4, sp, #3333

@ MOV reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
movge r4, r5
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movge r4, pc
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movge pc, r5
@ MOV reg, encoding T3 (32-bit) -- can only appear as MOVS or MOV.W
@ MOV imm, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
movge r4, #5
@ MOV imm, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movge r14, #5
@ MOV imm, encoding T3 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movge r14, #555

@ CMP reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
cmpge r3, r4
@ CMP reg, encoding T2
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
cmpge r13, r4
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
cmpge r3, pc
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
cmpge pc, r4
@ CMP reg, encoding T3 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
cmpge r3, r4, lsl #1 
@ CMP imm, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
cmpge r3, #4
@ CMP imm, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
cmpge r13, #4

@ AND reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
andge r5, r6
@ AND reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r9, r6

@ EOR reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
eorge r7, r6
@ EOR reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r7, r9

@ LSL imm, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
lslge r7, r0, #1
@ LSL imm, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
lslge r7, r10, #1
@ LSL reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
lslge r7, r0
@ LSL reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
lslge r7, r10

@ LSR imm, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
lsrge r3, r2, #1
@ LSR imm, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
lsrge r3, r12, #1
@ LSR reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
lsrge r3, r2
@ LSR reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
lsrge r3, r12

@ ASR imm, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
asrge r2, r3, #4
@ ASR imm, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
asrge r12, r3, #4
@ ASR reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
asrge r2, r3
@ ASR reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
asrge r12, r3

@ ADC reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
adcge r5, r4
@ ADC reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r5, r5, r14

@ SBC reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
sbcge r5, r6
@ SBC reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r9, r9, r6

@ ROR reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
rorge r7, r6
@ ROR reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rorge r7, r9

@ TST reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
tstge r7, r0
@ TST reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
tstge r7, r10

@ RSB imm, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
rsbge r1, r0, #0
@ RSB imm, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r11, r0, #0

@ CMN reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
cmnge r1, r2
@ CMN reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
cmnge r11, r2

@ ORR reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
orrge r3, r2
@ ORR reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r3, r12

@ MUL reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
mulge r3, r4, r3
@ MUL reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mulge r3, r4, r5

@ BIC reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
bicge r5, r4
@ BIC reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r5, r14

@ MVN reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
mvnge r5, r6
@ MVN reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mvnge r9, r6

@ BX, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
bxge r6
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bxge pc

@ BLX, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
blxge r7
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
blxge pc

@ LDR reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
ldrge r0, [r1, r2]
@ LDR reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge r10, [r1, r2]
@ LDR imm, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
ldrge r0, [r1]
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
ldrge r0, [r1, #8]
@ LDR imm, encoding T2
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
ldrge r0, [sp]
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
ldrge r0, [sp, #8]
@ LDR reg, encoding T3 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge r0, [r1, #2]
@ LDR reg, encoding T4 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge r0, [r1, #-2]
@ LDR lit, encoding T1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge r0, [pc, #8]
@ LDR lit, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge r10, [pc, #8]

@ STR reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
strge r1, [r2, r3]
@ STR reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge r11, [r2, r3]
@ STR imm, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
strge r1, [r2]
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
strge r1, [r2, #4]
@ STR imm, encoding T2
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
strge r1, [sp]
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
strge r1, [sp, #4]
@ STR imm, encoding T3 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge r1, [r2, #3]
@ STR imm, encoding T4 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge r1, [r2, #-3]

@ STRH reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
strhge r4, [r3, r2]
@ STRH reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge r14, [r3, r2]
@ STRH imm, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
strhge r4, [r3]
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
strhge r4, [r3, #2]
@ STRH imm, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge r4, [r3, #1]
@ STRH imm, encoding T3 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge r4, [r3, #-2]

@ STRB reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
strbge r3, [r4, r5]
@ STRB reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge r3, [r14, r5]
@ STRB imm, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
strbge r3, [r4]
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
strbge r3, [r4, #5]
@ STRB reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge r3, [r14, #5]
@ STRB reg, encoding T3 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge r3, [r4, #-5]

@ LDRSB reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
ldrsbge r6, [r5, r4]
@ LDRSB reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge r9, [r5, r4]

@ LDRH reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
ldrhge r5, [r6, r7]
@ LDRH reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge r5, [r9, r7]
@ LDRH imm, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
ldrhge r5, [r6]
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
ldrhge r5, [r6, #8]
@ LDRH imm, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge r5, [r6, #7]
@ LDRH imm, encoding T3 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge r5, [r6, #-8]

@ LDRB reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
ldrbge r0, [r7, r6]
@ LDRB reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge r10, [r7, r6]
@ LDRB imm, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
ldrbge r0, [r7]
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
ldrbge r0, [r7, #6]
@ LDRB reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge r10, [r7, #6]
@ LDRB reg, encoding T3 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge r0, [r7, #-6]

@ LDRSH reg, encoding T1
@ CHECK-NOT: :[[@LINE+2]]:1: warning
it ge
ldrshge r7, [r0, r1]
@ LDRSH reg, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge r7, [r0, r11]

@ ADR, encoding T1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adrge r1, #24
@ ADR, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adrge r1, #-23
@ ADR, encoding T3 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adrge r1, #23

@ SXTH, encoding T1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sxthge r4, r3
@ SXTH, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sxthge r4, r9

@ SXTB, encoding T1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sxtbge r4, r5
@ SXTB, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sxtbge r14, r5

@ UXTH, encoding T1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
uxthge r6, r5
@ UXTH, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
uxthge r9, r5

@ UXTB, encoding T1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
uxtbge r6, r7
@ UXTB, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
uxtbge r6, r9

@ PUSH, encoding T1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
pushge {r1, r3, r7}
@ PUSH, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
pushge {r1, r3, r7}
@ PUSH, encoding T3 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
pushge {r3}

@ REV, encoding T1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
revge r7, r6
@ REV, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
revge r9, r6

@ REV16, encoding T1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rev16ge r7, r0
@ REV16, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rev16ge r7, r10

@ REVSH, encoding T1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
revshge r1, r0
@ REVSH, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
revshge r11, r0

@ POP, encoding T1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
popge {r1, r0, r5}
@ POP, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
popge {r1, r5, r10}
@ POP, encoding T3 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
popge {r10}

@ NOP, encoding T1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
nopge
@ NOP, encoding T2 (32-bit) -- can only appear as NOP.W

@ STM, encoding T1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stmge r1!, {r2, r3}
@ STM, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stmge r1, {r2, r3}
@ STM, encoding T3 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stmge r1!, {r2, r3}

@ LDM, encoding T1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldmge r4!, {r2, r3}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldmge r4, {r2, r3}
@ LDM, encoding T2 (32-bit)
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldmge r14!, {r2, r3}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldmge r14, {r2, r3}

@ SVC, encoding T1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
svcge #55

@ B, encoding T2
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bge #2014

@ The following Thumb instructions only have 32-bit encodings.
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strexge r0, r0, [r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strexge r0, r0, [r1]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strexge r0, r0, [r2]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strexge r0, r0, [r3]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strexge r0, r0, [r4]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strexge r0, r0, [r5]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strexge r0, r0, [r6]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strexge r0, r0, [r7]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strexge r0, r0, [r8]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strexge r0, r0, [r9]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strexge r0, r0, [r10]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strexge r0, r0, [r11]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strexge r0, r0, [r12]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strexge r0, r0, [sp]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strexge r0, r0, [lr]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strexge r0, r0, [pc]
@ CHECK: [[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r0], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r1], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r2], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r3], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r4], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r5], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r6], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r7], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r8], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r9], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r10], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r11], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r12], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [sp], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [lr], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [pc], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r0], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r1], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r2], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r3], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r4], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r5], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r6], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r7], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r8], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r9], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r10], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r11], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r12], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [sp], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [lr], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [pc], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r0, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r1, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r2, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r3, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r4, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r5, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r6, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r7, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r8, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r9, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r10, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r11, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r12, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [sp, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [lr, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [pc, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r0, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r1, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r2, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r3, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r4, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r5, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r6, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r7, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r8, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r9, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r10, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r11, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r12, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [sp, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [lr, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [pc, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r1]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r2]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r3]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r4]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r5]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r6]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r7]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r8]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r9]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r10]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r11]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r12]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [sp]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [lr]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [pc]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r0, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r1, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r2, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r3, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r4, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r5, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r6, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r7, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r8, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r9, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r10, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r11, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [r12, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [sp, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [lr, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strdge r0, r0, [pc, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge.w r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge.w r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge.w r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge.w r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge.w r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge.w r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge.w r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge.w r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge.w r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge.w r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge.w r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge.w r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge.w r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge.w r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andsge.w r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andsge.w r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andsge.w r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andsge.w r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andsge.w r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andsge.w r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andsge.w r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andsge.w r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andsge.w r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andsge.w r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andsge.w r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andsge.w r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andsge.w r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andsge.w r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge.w r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge.w r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge.w r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge.w r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge.w r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge.w r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge.w r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge.w r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge.w r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge.w r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge.w r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge.w r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge.w r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge.w r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicsge.w r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicsge.w r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicsge.w r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicsge.w r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicsge.w r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicsge.w r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicsge.w r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicsge.w r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicsge.w r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicsge.w r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicsge.w r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicsge.w r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicsge.w r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicsge.w r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge.w r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge.w r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge.w r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge.w r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge.w r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge.w r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge.w r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge.w r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge.w r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge.w r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge.w r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge.w r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge.w r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge.w r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movge.w r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrsge.w r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrsge.w r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrsge.w r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrsge.w r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrsge.w r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrsge.w r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrsge.w r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrsge.w r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrsge.w r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrsge.w r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrsge.w r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrsge.w r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrsge.w r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrsge.w r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movsge.w r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mvnge.w r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mvnsge.w r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge.w r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge.w r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge.w r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge.w r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge.w r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge.w r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge.w r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge.w r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge.w r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge.w r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge.w r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge.w r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge.w r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge.w r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorsge.w r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorsge.w r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorsge.w r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorsge.w r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorsge.w r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorsge.w r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorsge.w r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorsge.w r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorsge.w r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorsge.w r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorsge.w r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorsge.w r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorsge.w r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorsge.w r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, sp, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, sp, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge.w r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge.w r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge.w r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge.w r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge.w r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge.w r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge.w r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge.w r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge.w r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge.w r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge.w r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge.w r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge.w r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge.w r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcsge.w r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcsge.w r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcsge.w r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcsge.w r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcsge.w r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcsge.w r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcsge.w r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcsge.w r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcsge.w r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcsge.w r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcsge.w r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcsge.w r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcsge.w r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcsge.w r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge.w r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge.w r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge.w r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge.w r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge.w r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge.w r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge.w r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge.w r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge.w r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge.w r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge.w r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge.w r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge.w r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge.w r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcsge.w r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcsge.w r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcsge.w r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcsge.w r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcsge.w r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcsge.w r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcsge.w r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcsge.w r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcsge.w r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcsge.w r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcsge.w r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcsge.w r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcsge.w r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcsge.w r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, sp, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, sp, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r0], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r1], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r2], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r3], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r4], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r5], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r6], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r7], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r8], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r9], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r10], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r11], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r12], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [sp], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [lr], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [pc], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r0], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r1], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r2], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r3], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r4], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r5], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r6], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r7], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r8], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r9], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r10], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r11], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r12], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [sp], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [lr], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [pc], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mcrrge p0, #0, r0, r0, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mcrrge p0, #0, r0, r1, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mcrrge p0, #0, r0, r2, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mcrrge p0, #0, r0, r3, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mcrrge p0, #0, r0, r4, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mcrrge p0, #0, r0, r5, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mcrrge p0, #0, r0, r6, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mcrrge p0, #0, r0, r7, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mcrrge p0, #0, r0, r8, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mcrrge p0, #0, r0, r9, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mcrrge p0, #0, r0, r10, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mcrrge p0, #0, r0, r11, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mcrrge p0, #0, r0, r12, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mcrrge p0, #0, r0, sp, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mcrrge p0, #0, r0, lr, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mcrrge p0, #0, r0, pc, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mrrcge p14, #0, r0, r0, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mrrcge p14, #0, r0, r1, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mrrcge p14, #0, r0, r2, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mrrcge p14, #0, r0, r3, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mrrcge p14, #0, r0, r4, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mrrcge p14, #0, r0, r5, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mrrcge p14, #0, r0, r6, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mrrcge p14, #0, r0, r7, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mrrcge p14, #0, r0, r8, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mrrcge p14, #0, r0, r9, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mrrcge p14, #0, r0, r10, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mrrcge p14, #0, r0, r11, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mrrcge p14, #0, r0, r12, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mrrcge p14, #0, r0, sp, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mrrcge p14, #0, r0, lr, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mrrcge p14, #0, r0, pc, c0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r0], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r1], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r2], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r3], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r4], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r5], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r6], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r7], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r8], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r9], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r10], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r11], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r12], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [sp], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [lr], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [pc], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r0], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r1], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r2], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r3], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r4], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r5], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r6], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r7], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r8], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r9], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r10], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r11], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r12], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [sp], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [lr], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [pc], #-0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r0], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r1], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r2], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r3], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r4], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r5], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r6], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r7], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r8], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r9], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r10], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r11], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r12], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [sp], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [lr], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [pc], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r0], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r1], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r2], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r3], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r4], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r5], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r6], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r7], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r8], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r9], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r10], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r11], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r12], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [sp], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [lr], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [pc], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r0], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r1], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r2], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r3], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r4], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r5], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r6], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r7], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r8], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r9], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r10], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r11], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r12], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [sp], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [lr], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [pc], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r0], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r1], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r2], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r3], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r4], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r5], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r6], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r7], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r8], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r9], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r10], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r11], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r12], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [sp], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [lr], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [pc], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r0], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r1], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r2], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r3], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r4], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r5], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r6], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r7], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r8], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r9], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r10], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r11], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r12], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [sp], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [lr], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [pc], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r0], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r1], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r2], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r3], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r4], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r5], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r6], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r7], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r8], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r9], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r10], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r11], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r12], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [sp], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [lr], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [pc], {0}
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r0], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r1], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r2], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r3], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r4], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r5], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r6], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r7], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r8], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r9], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r10], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r11], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r12], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [sp], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [lr], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [pc], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r0], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r1], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r2], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r3], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r4], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r5], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r6], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r7], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r8], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r9], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r10], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r11], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r12], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [sp], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [lr], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [pc], #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r0, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r1, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r2, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r3, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r4, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r5, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r6, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r7, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r8, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r9, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r10, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r11, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r12, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [sp, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [lr, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [pc, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r0, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r1, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r2, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r3, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r4, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r5, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r6, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r7, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r8, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r9, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r10, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r11, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r12, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [sp, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [lr, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [pc, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r0, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r1, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r2, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r3, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r4, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r5, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r6, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r7, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r8, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r9, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r10, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r11, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r12, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [sp, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [lr, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [pc, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r0, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r1, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r2, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r3, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r4, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r5, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r6, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r7, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r8, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r9, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r10, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r11, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r12, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [sp, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [lr, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [pc, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r0, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r1, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r2, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r3, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r4, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r5, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r6, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r7, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r8, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r9, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r10, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r11, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r12, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [sp, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [lr, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [pc, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r0, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r1, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r2, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r3, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r4, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r5, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r6, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r7, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r8, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r9, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r10, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r11, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r12, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [sp, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [lr, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [pc, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r0, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r1, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r2, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r3, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r4, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r5, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r6, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r7, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r8, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r9, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r10, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r11, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r12, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [sp, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [lr, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [pc, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r0, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r1, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r2, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r3, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r4, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r5, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r6, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r7, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r8, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r9, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r10, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r11, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r12, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [sp, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [lr, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [pc, #-0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r1]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r2]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r3]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r4]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r5]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r6]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r7]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r8]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r9]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r10]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r11]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r12]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [sp]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [lr]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [pc]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r1]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r2]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r3]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r4]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r5]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r6]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r7]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r8]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r9]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r10]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r11]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r12]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [sp]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [lr]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [pc]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r0, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r1, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r2, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r3, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r4, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r5, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r6, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r7, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r8, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r9, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r10, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r11, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [r12, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [sp, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [lr, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stcge p0, c0, [pc, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r0, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r1, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r2, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r3, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r4, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r5, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r6, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r7, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r8, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r9, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r10, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r11, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [r12, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [sp, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [lr, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldcge p0, c0, [pc, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r1]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r2]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r3]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r4]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r5]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r6]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r7]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r8]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r9]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r10]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r11]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r12]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [sp]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [lr]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [pc]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r1]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r2]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r3]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r4]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r5]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r6]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r7]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r8]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r9]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r10]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r11]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r12]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [sp]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [lr]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [pc]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r0, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r1, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r2, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r3, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r4, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r5, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r6, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r7, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r8, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r9, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r10, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r11, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [r12, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [sp, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [lr, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
stclge p0, c0, [pc, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r0, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r1, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r2, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r3, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r4, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r5, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r6, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r7, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r8, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r9, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r10, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r11, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [r12, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [sp, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [lr, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldclge p0, c0, [pc, #0]!
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r1, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r2, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r3, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r4, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r5, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r6, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r7, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r8, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r9, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r10, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r11, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r12, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, lr, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r1, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r2, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r3, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r4, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r5, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r6, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r7, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r8, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r9, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r10, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r11, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r12, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, lr, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r1, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r2, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r3, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r4, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r5, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r6, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r7, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r8, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r9, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r10, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r11, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r12, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, lr, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r1, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r2, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r3, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r4, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r5, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r6, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r7, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r8, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r9, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r10, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r11, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r12, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, lr, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r1, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r2, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r3, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r4, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r5, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r6, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r7, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r8, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r9, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r10, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r11, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r12, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, lr, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movge.w r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r1, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r2, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r3, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r4, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r5, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r6, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r7, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r8, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r9, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r10, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r11, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r12, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, lr, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movsge.w r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r1, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r2, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r3, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r4, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r5, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r6, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r7, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r8, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r9, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r10, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r11, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r12, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, lr, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mvnge r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r1, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r2, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r3, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r4, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r5, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r6, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r7, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r8, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r9, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r10, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r11, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r12, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, lr, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mvnge r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r1, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r2, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r3, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r4, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r5, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r6, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r7, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r8, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r9, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r10, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r11, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r12, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, lr, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r1, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r2, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r3, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r4, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r5, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r6, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r7, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r8, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r9, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r10, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r11, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r12, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, lr, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r1, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r2, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r3, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r4, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r5, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r6, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r7, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r8, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r9, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r10, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r11, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r12, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, sp, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, lr, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r1, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r2, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r3, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r4, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r5, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r6, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r7, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r8, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r9, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r10, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r11, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r12, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, sp, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, lr, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r1, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r2, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r3, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r4, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r5, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r6, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r7, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r8, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r9, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r10, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r11, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r12, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, lr, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r1, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r2, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r3, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r4, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r5, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r6, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r7, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r8, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r9, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r10, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r11, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r12, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, lr, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r1, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r2, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r3, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r4, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r5, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r6, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r7, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r8, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r9, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r10, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r11, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r12, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, lr, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r1, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r2, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r3, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r4, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r5, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r6, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r7, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r8, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r9, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r10, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r11, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r12, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, lr, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r1, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r2, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r3, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r4, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r5, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r6, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r7, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r8, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r9, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r10, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r11, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r12, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, sp, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, lr, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r1, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r2, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r3, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r4, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r5, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r6, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r7, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r8, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r9, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r10, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r11, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r12, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, sp, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, lr, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r1, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r2, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r3, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r4, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r5, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r6, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r7, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r8, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r9, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r10, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r11, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r12, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, lr, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r1, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r2, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r3, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r4, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r5, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r6, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r7, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r8, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r9, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r10, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r11, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r12, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, lr, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r1, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r2, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r3, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r4, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r5, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r6, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r7, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r8, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r9, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r10, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r11, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r12, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, sp, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, lr, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, pc, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #4096
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #8192
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #12288
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #16384
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #20480
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #24576
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #28672
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #32768
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #36864
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #40960
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #45056
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #49152
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #53248
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #57344
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #61440
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r1, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r2, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r3, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r4, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r5, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r6, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r7, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r8, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r9, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r10, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r11, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r12, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, sp, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, lr, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, pc, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #4096
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #8192
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #12288
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #16384
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #20480
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #24576
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #28672
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #32768
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #36864
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #40960
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #45056
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #49152
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #53248
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #57344
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #61440
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r2
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r3
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r4
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r5
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r6
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r7
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r8
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r9
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r10
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r11
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r12
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, lr
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r0, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r1, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r2, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r3, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r4, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r5, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r6, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r7, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r8, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r9, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r10, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r11, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r12, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, lr, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bfige r0, r0, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bfige r0, r1, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bfige r0, r2, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bfige r0, r3, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bfige r0, r4, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bfige r0, r5, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bfige r0, r6, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bfige r0, r7, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bfige r0, r8, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bfige r0, r9, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bfige r0, r10, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bfige r0, r11, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bfige r0, r12, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bfige r0, lr, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bfcge r0, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r2
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r3
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r4
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r5
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r6
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r7
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r8
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r9
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r10
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r11
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r12
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, lr
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r0, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r1, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r2, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r3, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r4, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r5, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r6, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r7, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r8, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r9, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r10, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r11, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r12, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, lr, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r1, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r2, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r3, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r4, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r5, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r6, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r7, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r8, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r9, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r10, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r11, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r12, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, lr, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r1, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r2, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r3, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r4, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r5, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r6, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r7, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r8, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r9, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r10, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r11, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, r12, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
andge r0, lr, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r1, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r2, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r3, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r4, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r5, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r6, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r7, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r8, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r9, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r10, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r11, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r12, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, lr, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r1, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r2, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r3, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r4, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r5, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r6, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r7, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r8, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r9, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r10, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r11, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, r12, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
bicge r0, lr, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r1, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r2, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r3, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r4, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r5, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r6, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r7, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r8, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r9, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r10, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r11, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r12, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, lr, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movge.w r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r1, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r2, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r3, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r4, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r5, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r6, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r7, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r8, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r9, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r10, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r11, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, r12, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
orrge r0, lr, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movsge.w r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r1, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r2, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r3, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r4, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r5, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r6, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r7, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r8, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r9, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r10, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r11, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r12, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, lr, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mvnge r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r1, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r2, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r3, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r4, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r5, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r6, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r7, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r8, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r9, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r10, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r11, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, r12, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ornge r0, lr, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mvnge r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r1, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r2, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r3, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r4, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r5, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r6, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r7, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r8, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r9, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r10, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r11, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r12, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, lr, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r1, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r2, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r3, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r4, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r5, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r6, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r7, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r8, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r9, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r10, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r11, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, r12, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
eorge r0, lr, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r1, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r2, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r3, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r4, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r5, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r6, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r7, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r8, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r9, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r10, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r11, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, r12, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, sp, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addge.w r0, lr, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r1, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r2, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r3, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r4, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r5, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r6, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r7, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r8, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r9, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r10, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r11, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, r12, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, sp, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addsge.w r0, lr, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r1, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r2, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r3, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r4, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r5, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r6, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r7, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r8, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r9, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r10, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r11, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r12, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, lr, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r1, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r2, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r3, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r4, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r5, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r6, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r7, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r8, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r9, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r10, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r11, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, r12, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
adcge r0, lr, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r1, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r2, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r3, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r4, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r5, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r6, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r7, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r8, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r9, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r10, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r11, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r12, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, lr, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r1, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r2, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r3, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r4, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r5, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r6, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r7, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r8, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r9, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r10, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r11, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, r12, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbcge r0, lr, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r1, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r2, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r3, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r4, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r5, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r6, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r7, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r8, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r9, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r10, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r11, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, r12, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, sp, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subge.w r0, lr, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r1, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r2, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r3, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r4, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r5, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r6, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r7, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r8, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r9, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r10, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r11, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, r12, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, sp, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subsge.w r0, lr, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r1, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r2, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r3, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r4, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r5, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r6, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r7, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r8, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r9, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r10, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r11, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, r12, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbge.w r0, lr, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r0, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r1, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r2, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r3, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r4, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r5, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r6, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r7, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r8, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r9, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r10, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r11, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, r12, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
rsbsge.w r0, lr, #8388608
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r0, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r1, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r2, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r3, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r4, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r5, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r6, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r7, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r8, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r9, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r10, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r11, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, r12, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, sp, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, lr, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
addwge r0, pc, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #6144
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #10240
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #14336
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #18432
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #22528
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #26624
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #30720
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #34816
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #38912
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #43008
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #47104
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #51200
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #55296
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #59392
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movwge r0, #63488
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r0, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r1, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r2, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r3, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r4, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r5, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r6, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r7, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r8, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r9, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r10, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r11, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, r12, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, sp, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, lr, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
subwge r0, pc, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #2048
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #6144
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #10240
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #14336
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #18432
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #22528
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #26624
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #30720
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #34816
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #38912
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #43008
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #47104
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #51200
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #55296
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #59392
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
movtge r0, #63488
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r2
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r3
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r4
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r5
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r6
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r7
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r8
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r9
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r10
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r11
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, r12
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ssatge r0, #1, lr
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r0, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r1, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r2, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r3, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r4, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r5, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r6, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r7, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r8, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r9, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r10, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r11, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, r12, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
sbfxge r0, lr, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r2
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r3
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r4
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r5
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r6
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r7
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r8
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r9
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r10
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r11
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, r12
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
usatge r0, #0, lr
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r0, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r1, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r2, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r3, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r4, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r5, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r6, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r7, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r8, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r9, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r10, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r11, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, r12, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ubfxge r0, lr, #0, #1
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r0, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r1, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r2, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r3, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r4, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r5, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r6, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r7, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r8, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r9, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r10, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r11, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r12, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [sp, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [lr, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r0, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r1, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r2, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r3, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r4, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r5, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r6, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r7, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r8, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r9, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r10, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r11, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r12, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [sp, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [lr, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [pc, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r0, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r1, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r2, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r3, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r4, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r5, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r6, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r7, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r8, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r9, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r10, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r11, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r12, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [sp, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [lr, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r0, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r1, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r2, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r3, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r4, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r5, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r6, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r7, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r8, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r9, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r10, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r11, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r12, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [sp, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [lr, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [pc, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r0, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r1, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r2, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r3, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r4, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r5, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r6, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r7, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r8, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r9, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r10, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r11, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r12, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [sp, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [lr, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r0, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r1, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r2, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r3, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r4, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r5, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r6, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r7, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r8, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r9, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r10, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r11, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r12, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [sp, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [lr, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [pc, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r1]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r2]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r3]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r4]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r5]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r6]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r7]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r8]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r9]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r10]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r11]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [r12]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [sp]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strbge.w r0, [lr]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r1]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r2]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r3]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r4]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r5]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r6]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r7]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r8]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r9]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r10]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r11]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [r12]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [sp]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [lr]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrbge.w r0, [pc, #0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r1]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r2]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r3]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r4]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r5]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r6]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r7]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r8]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r9]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r10]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r11]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [r12]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [sp]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strhge.w r0, [lr]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r1]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r2]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r3]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r4]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r5]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r6]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r7]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r8]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r9]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r10]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r11]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [r12]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [sp]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [lr]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrhge.w r0, [pc, #0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r1]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r2]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r3]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r4]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r5]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r6]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r7]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r8]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r9]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r10]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r11]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [r12]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [sp]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
strge.w r0, [lr]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r1]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r2]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r3]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r4]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r5]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r6]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r7]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r8]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r9]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r10]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r11]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [r12]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [sp]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [lr]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrge.w r0, [pc, #0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r0, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r1, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r2, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r3, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r4, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r5, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r6, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r7, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r8, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r9, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r10, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r11, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r12, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [sp, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [lr, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [pc, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r0, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r1, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r2, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r3, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r4, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r5, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r6, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r7, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r8, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r9, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r10, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r11, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r12, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [sp, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [lr, r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [pc, #-0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r1]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r2]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r3]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r4]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r5]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r6]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r7]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r8]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r9]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r10]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r11]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [r12]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [sp]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [lr]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrsbge.w r0, [pc, #0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r1]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r2]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r3]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r4]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r5]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r6]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r7]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r8]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r9]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r10]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r11]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [r12]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [sp]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [lr]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
ldrshge.w r0, [pc, #0]
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mlage r0, r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mlage r0, r1, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mlage r0, r2, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mlage r0, r3, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mlage r0, r4, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mlage r0, r5, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mlage r0, r6, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mlage r0, r7, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mlage r0, r8, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mlage r0, r9, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mlage r0, r10, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mlage r0, r11, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mlage r0, r12, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
mlage r0, lr, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smullge r0, r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smullge r0, r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smullge r0, r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smullge r0, r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smullge r0, r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smullge r0, r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smullge r0, r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smullge r0, r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smullge r0, r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smullge r0, r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smullge r0, r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smullge r0, r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smullge r0, r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smullge r0, r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umullge r0, r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umullge r0, r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umullge r0, r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umullge r0, r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umullge r0, r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umullge r0, r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umullge r0, r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umullge r0, r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umullge r0, r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umullge r0, r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umullge r0, r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umullge r0, r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umullge r0, r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umullge r0, r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smlalge r0, r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smlalge r0, r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smlalge r0, r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smlalge r0, r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smlalge r0, r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smlalge r0, r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smlalge r0, r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smlalge r0, r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smlalge r0, r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smlalge r0, r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smlalge r0, r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smlalge r0, r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smlalge r0, r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
smlalge r0, r0, lr, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umlalge r0, r0, r0, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umlalge r0, r0, r1, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umlalge r0, r0, r2, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umlalge r0, r0, r3, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umlalge r0, r0, r4, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umlalge r0, r0, r5, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umlalge r0, r0, r6, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umlalge r0, r0, r7, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umlalge r0, r0, r8, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umlalge r0, r0, r9, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umlalge r0, r0, r10, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umlalge r0, r0, r11, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umlalge r0, r0, r12, r0
@ CHECK: :[[@LINE+2]]:1: warning: deprecated instruction in IT block
it ge
umlalge r0, r0, lr, r0
