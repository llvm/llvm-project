REQUIRES: aarch64-registered-target

## PPR Register Class Initialization Testcase
## Ideally, we should use PTRUE_{B/H/S/D} instead of FADDV_VPZ_D for an isolated test case; 
## However, exegesis does not yet support PTRUE_{B/H/S/D}.
RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%d --opcode-name=FADDV_VPZ_D --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %d > %t.s
RUN: FileCheck %s --check-prefix=PPR_ASM < %t.s
PPR_ASM:            <foo>:
PPR_ASM:            ptrue p{{[0-9]+}}.b
PPR_ASM-NEXT:       mov z{{[0-9]+}}.d, #0x0
PPR_ASM-NEXT:       faddv d{{[0-9]+}}, p{{[0-9]+}}, z{{[0-9]+}}

## ZPR Register Class Initialization Testcase
## Ideally, we should use DUP_ZI_{B/H/S/D} instead of FADDV_VPZ_D for an isolated test case; 
## However, exegesis does not yet support DUP_ZI_{B/H/S/D}.
RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%d --opcode-name=FADDV_VPZ_D --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %d > %t.s
RUN: FileCheck %s --check-prefix=ZPR_ASM < %t.s
ZPR_ASM:            <foo>:
ZPR_ASM:            ptrue p{{[0-9]+}}.b
ZPR_ASM-NEXT:       mov z{{[0-9]+}}.d, #0x0
ZPR_ASM-NEXT:       faddv d{{[0-9]+}}, p{{[0-9]+}}, z{{[0-9]+}}

## FPR128 Register Class Initialization Testcase
RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%d --opcode-name=ADDVv16i8v --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %d > %t.s
RUN: FileCheck %s --check-prefix=FPR128-ASM < %t.s
FPR128-ASM:         <foo>:
FPR128-ASM:         movi v{{[0-9]+}}.2d, #0000000000000000
FPR128-ASM-NEXT:    addv b{{[0-9]+}}, v{{[0-9]+}}.16b

## FPR64 Register Class Initialization Testcase
RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%d --opcode-name=ADDVv4i16v --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %d > %t.s
RUN: FileCheck %s --check-prefix=FPR64-ASM < %t.s
FPR64-ASM:          <foo>:
FPR64-ASM:          movi d{{[0-9]+}}, #0000000000000000
FPR64-ASM-NEXT:     addv h{{[0-9]+}}, v{{[0-9]+}}.4h

## FPR32 Register Class Initialization Testcase
RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%d --opcode-name=FABSSr --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %d > %t.s
RUN: FileCheck %s --check-prefix=FPR32-ASM < %t.s
FPR32-ASM:         <foo>:
FPR32-ASM:         movi d{{[0-9]+}}, #0000000000000000
FPR32-ASM-NEXT:    fabs s{{[0-9]+}}, s{{[0-9]+}}


## FPR16 Register Class Initialization Testcase
RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%d --opcode-name=FABSHr --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %d > %t.s
RUN: FileCheck %s --check-prefix=FPR16-ASM < %t.s
FPR16-ASM:         <foo>:
FPR16-ASM:         movi d{{[0-9]+}}, #0000000000000000
FPR16-ASM-NEXT:    fabs h{{[0-9]+}}, h{{[0-9]+}}

## FPR8 Register Class Initialization Testcase
RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%d --opcode-name=SQABSv1i8 --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %d > %t.s
RUN: FileCheck %s --check-prefix=FPR8-ASM < %t.s
FPR8-ASM:         <foo>:
FPR8-ASM:         movi    d{{[0-9]+}}, #0000000000000000
FPR8-ASM-NEXT:    sqabs   b{{[0-9]+}}, b{{[0-9]+}}


## FPCR Register Class Initialization Testcase
RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%d --opcode-name=BFCVT --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %d > %t.s
RUN: FileCheck %s --check-prefix=FPCR-ASM < %t.s
FPCR-ASM:         <foo>:
FPCR-ASM:         movi    d{{[0-9]+}}, #0000000000000000
FPCR-ASM-NEXT:    mov     x8, #0x0
FPCR-ASM-NEXT:    msr     FPCR, x8
FPCR-ASM-NEXT:    bfcvt   h{{[0-9]+}}, s{{[0-9]+}}
