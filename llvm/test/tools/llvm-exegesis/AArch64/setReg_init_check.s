REQUIRES: aarch64-registered-target

## PPR Register Class Initialization Testcase
## Ideally, we should use PTRUE_{B/H/S/D} instead of FADDV_VPZ_D for an isolated test case; 
## However, exegesis does not yet support PTRUE_{B/H/S/D}.
RUN: llvm-exegesis -mtriple=aarch64 -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%t.o --opcode-name=FADDV_VPZ_D --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %t.o > %t.s
RUN: FileCheck %s --check-prefix=PPR_ASM < %t.s
PPR_ASM:            <foo>:
PPR_ASM:            ptrue p{{[0-9]+}}.b
PPR_ASM-NEXT:       mov z{{[0-9]+}}.d, #0x0
PPR_ASM-NEXT:       faddv d{{[0-9]+}}, p{{[0-9]+}}, z{{[0-9]+}}

## ZPR Register Class Initialization Testcase
## Ideally, we should use DUP_ZI_{B/H/S/D} instead of FADDV_VPZ_D for an isolated test case; 
## However, exegesis does not yet support DUP_ZI_{B/H/S/D}.
RUN: llvm-exegesis -mtriple=aarch64 -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%t.o --opcode-name=FADDV_VPZ_D --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %t.o > %t.s
RUN: FileCheck %s --check-prefix=ZPR_ASM < %t.s
ZPR_ASM:            <foo>:
ZPR_ASM:            ptrue p{{[0-9]+}}.b
ZPR_ASM-NEXT:       mov z{{[0-9]+}}.d, #0x0
ZPR_ASM-NEXT:       faddv d{{[0-9]+}}, p{{[0-9]+}}, z{{[0-9]+}}

## FPR128 Register Class Initialization Testcase
RUN: llvm-exegesis -mtriple=aarch64 -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%t.o --opcode-name=ADDVv16i8v --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %t.o > %t.s
RUN: FileCheck %s --check-prefix=FPR128-ASM < %t.s
FPR128-ASM:         <foo>:
FPR128-ASM:         movi v{{[0-9]+}}.2d, #0000000000000000
FPR128-ASM-NEXT:    addv b{{[0-9]+}}, v{{[0-9]+}}.16b

## FPR64 Register Class Initialization Testcase
RUN: llvm-exegesis -mtriple=aarch64 -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%t.o --opcode-name=ADDVv4i16v --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %t.o > %t.s
RUN: FileCheck %s --check-prefix=FPR64-ASM < %t.s
FPR64-ASM:          <foo>:
FPR64-ASM:          movi d{{[0-9]+}}, #0000000000000000
FPR64-ASM-NEXT:     addv h{{[0-9]+}}, v{{[0-9]+}}.4h

## FPR32 Register Class Initialization Testcase
RUN: llvm-exegesis -mtriple=aarch64 -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%t.o --opcode-name=FABSSr --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %t.o > %t.s
RUN: FileCheck %s --check-prefix=FPR32-ASM < %t.s
FPR32-ASM:         <foo>:
FPR32-ASM:         movi d{{[0-9]+}}, #0000000000000000
FPR32-ASM-NEXT:    fabs s{{[0-9]+}}, s{{[0-9]+}}


## FPR16 Register Class Initialization Testcase
RUN: llvm-exegesis -mtriple=aarch64 -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%t.o --opcode-name=FABSHr --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %t.o > %t.s
RUN: FileCheck %s --check-prefix=FPR16-ASM < %t.s
FPR16-ASM:         <foo>:
FPR16-ASM:         movi d{{[0-9]+}}, #0000000000000000
FPR16-ASM-NEXT:    fabs h{{[0-9]+}}, h{{[0-9]+}}

## FPR8 Register Class Initialization Testcase
RUN: llvm-exegesis -mtriple=aarch64 -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%t.o --opcode-name=SQABSv1i8 --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %t.o > %t.s
RUN: FileCheck %s --check-prefix=FPR8-ASM < %t.s
FPR8-ASM:         <foo>:
FPR8-ASM:         movi    d{{[0-9]+}}, #0000000000000000
FPR8-ASM-NEXT:    sqabs   b{{[0-9]+}}, b{{[0-9]+}}


## FPCR Register Class Initialization Testcase
RUN: llvm-exegesis -mtriple=aarch64 -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%t.o --opcode-name=BFCVT --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %t.o > %t.s
RUN: FileCheck %s --check-prefix=FPCR-ASM < %t.s
FPCR-ASM:         <foo>:
FPCR-ASM:         movi    d{{[0-9]+}}, #0000000000000000
FPCR-ASM-NEXT:    mov     x8, #0x0
FPCR-ASM-NEXT:    msr     FPCR, x8
FPCR-ASM-NEXT:    bfcvt   h{{[0-9]+}}, s{{[0-9]+}}

## NZCV Register Class Initialization Testcase
RUN: llvm-exegesis -mtriple=aarch64 -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%t.o --opcode-name=ADCSWr --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %t.o > %t.s
RUN: FileCheck %s --check-prefix=NZCV-ASM < %t.s
NZCV-ASM:         <foo>:
NZCV-ASM:         mrs     x8, NZCV
NZCV-ASM-NEXT:    mov     x9, #0xf0000000
NZCV-ASM-NEXT:    bic     x8, x8, x9
NZCV-ASM-NEXT:    msr     NZCV, x8
NZCV-ASM-NEXT:    adcs    w{{[0-9]+}}, w{{[0-9]+}}, w{{[0-9]+}}

## FFR Register Class Initialization Testcase
RUN: llvm-exegesis -mtriple=aarch64 -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%t.o --opcode-name=GLDFF1B_D --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %t.o > %t.s
RUN: FileCheck %s --check-prefix=FFR-ASM < %t.s
FFR-ASM:         <foo>:
FFR-ASM:         setffr
FFR-ASM-NEXT:    ldff1b    { z{{[0-9]+}}.d }, p{{[0-9]+}}/z, [x{{[0-9]+}}, z{{[0-9]+}}.d]

# NOTE: Multi-register classes below can wrap around, e.g. { v31.1d, v0.1d }
# FileCheck cant handle this constraint and '--random-generator-seed=<seed>'
# can't be used either as the implementation relies on uniform_int_distribution
# which produces inconsistent results across libc++ implementations.

## WSeqPair Register Class Initialization Testcase
RUN: llvm-exegesis -mtriple=aarch64 -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%t.o --opcode-name=CASPW --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %t.o > %t.s
RUN: FileCheck %s --check-prefix=WSEQPAIR-ASM < %t.s
WSEQPAIR-ASM:         <foo>:
WSEQPAIR-ASM:         mov     w{{[0-9]+}}, #0x0
WSEQPAIR-ASM-NEXT:    mov     w{{[0-9]+}}, #0x0
WSEQPAIR-ASM:         casp    w{{[0-9]+}}, w{{[0-9]+}}, w{{[0-9]+}}, w{{[0-9]+}}, [x{{[0-9]+}}]

## XSeqPair Register Class Initialization Testcase
RUN: llvm-exegesis -mtriple=aarch64 -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%t.o --opcode-name=CASPX --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %t.o > %t.s
RUN: FileCheck %s --check-prefix=XSEQPAIR-ASM < %t.s
XSEQPAIR-ASM:         <foo>:
XSEQPAIR-ASM:         mov     x{{[0-9]+}}, #0x{{[0-9]+}}
XSEQPAIR-ASM-NEXT:    mov     x{{[0-9]+}}, #0x{{[0-9]+}}
XSEQPAIR-ASM:         casp    x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}, [x{{[0-9]+}}]

## DD Register Class Initialization Testcase
RUN: llvm-exegesis -mtriple=aarch64 -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%t.o --opcode-name=ST1Twov1d_POST --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %t.o > %t.s
RUN: FileCheck %s --check-prefix=DD-ASM < %t.s
DD-ASM:         <foo>:
DD-ASM:         movi     d{{[0-9]+}}, #0000000000000000
DD-ASM-NEXT:    movi     d{{[0-9]+}}, #0000000000000000
DD-ASM:         st1      { v{{[0-9]+}}.1d, v{{[0-9]+}}.1d }, [x{{[0-9]+}}], x{{[0-9]+}}

## DDD Register Class Initialization Testcase
RUN: llvm-exegesis -mtriple=aarch64 -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%t.o --opcode-name=ST1Threev1d_POST --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %t.o > %t.s
RUN: FileCheck %s --check-prefix=DDD-ASM < %t.s
DDD-ASM:         <foo>:
DDD-ASM:         movi     d{{[0-9]+}}, #0000000000000000
DDD-ASM-NEXT:    movi     d{{[0-9]+}}, #0000000000000000
DDD-ASM-NEXT:    movi     d{{[0-9]+}}, #0000000000000000
DDD-ASM:         st1      { v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d }, [x{{[0-9]+}}], x{{[0-9]+}}

## DDDD Register Class Initialization Testcase
RUN: llvm-exegesis -mtriple=aarch64 -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%t.o --opcode-name=ST1Fourv1d_POST --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %t.o > %t.s
RUN: FileCheck %s --check-prefix=DDDD-ASM < %t.s
DDDD-ASM:         <foo>:
DDDD-ASM:         movi     d{{[0-9]+}}, #0000000000000000
DDDD-ASM-NEXT:    movi     d{{[0-9]+}}, #0000000000000000
DDDD-ASM-NEXT:    movi     d{{[0-9]+}}, #0000000000000000
DDDD-ASM-NEXT:    movi     d{{[0-9]+}}, #0000000000000000
DDDD-ASM:         st1      { v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d }, [x{{[0-9]+}}], x{{[0-9]+}}

## QQ Register Class Initialization Testcase
RUN: llvm-exegesis -mtriple=aarch64 -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%t.o --opcode-name=ST1Twov16b_POST --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %t.o > %t.s
RUN: FileCheck %s --check-prefix=QQ-ASM < %t.s
QQ-ASM:         <foo>:
QQ-ASM:         movi     v{{[0-9]+}}.2d, #0000000000000000
QQ-ASM-NEXT:    movi     v{{[0-9]+}}.2d, #0000000000000000
QQ-ASM:         st1      { v{{[0-9]+}}.16b, v{{[0-9]+}}.16b }, [x{{[0-9]+}}], x{{[0-9]+}}

## QQQ Register Class Initialization Testcase
RUN: llvm-exegesis -mtriple=aarch64 -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%t.o --opcode-name=ST1Threev16b_POST --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %t.o > %t.s
RUN: FileCheck %s --check-prefix=QQQ-ASM < %t.s
QQQ-ASM:         <foo>:
QQQ-ASM:         movi     v{{[0-9]+}}.2d, #0000000000000000
QQQ-ASM-NEXT:    movi     v{{[0-9]+}}.2d, #0000000000000000
QQQ-ASM-NEXT:    movi     v{{[0-9]+}}.2d, #0000000000000000
QQQ-ASM:         st1      { v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b }, [x{{[0-9]+}}], x{{[0-9]+}}

## QQQQ Register Class Initialization Testcase
RUN: llvm-exegesis -mtriple=aarch64 -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%t.o --opcode-name=ST1Fourv16b_POST --benchmark-phase=assemble-measured-code 2>&1
RUN: llvm-objdump -d %t.o > %t.s
RUN: FileCheck %s --check-prefix=QQQQ-ASM < %t.s
QQQQ-ASM:         <foo>:
QQQQ-ASM:         movi     v{{[0-9]+}}.2d, #0000000000000000
QQQQ-ASM-NEXT:    movi     v{{[0-9]+}}.2d, #0000000000000000
QQQQ-ASM-NEXT:    movi     v{{[0-9]+}}.2d, #0000000000000000
QQQQ-ASM-NEXT:    movi     v{{[0-9]+}}.2d, #0000000000000000
QQQQ-ASM:         st1      { v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b }, [x{{[0-9]+}}], x{{[0-9]+}}
