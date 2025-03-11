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
