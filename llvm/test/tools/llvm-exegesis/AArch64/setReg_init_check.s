; REQUIRES: aarch64-registered-target

# PPR REGISTER CLASS INITIALIZATION TESTCASE 
# IDEALLY WE SHOULD USE PTRUE_{B/H?S/D} INSTEAD OF FADDV_VPZ_D FOR ISOLATED TESTCASE; BUT EXEGESIS DOES NOT SUPPORT PTRUE_{B/H?S/D} YET;
; RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%d --opcode-name=FADDV_VPZ_D 2>&1 | FileCheck %s --check-prefix=PPR
; RUN: llvm-objdump -d %d > %t.s
; RUN: FileCheck %s --check-prefix=PPR_ASM < %t.s
; PPR-NOT: setRegTo is not implemented, results will be unreliable
; PPR: assembled_snippet: {{.*}}C0035FD6
; PPR_ASM: {{<foo>:}}
; PPR_ASM: ptrue p{{[0-9]+}}.b
; PPR_ASM-NEXT: dupm z{{[0-9]+}}.s, #0x1
; PPR_ASM-NEXT: faddv d{{[0-9]+}}, p{{[0-9]+}}, z{{[0-9]+}}

# ZPR REGISTER CLASS INITIALIZATION TESTCASE 
# IDEALLY WE SHOULD USE PTRUE_{B/H?S/D} INSTEAD OF FADDV_VPZ_D FOR ISOLATED TESTCASE; BUT EXEGESIS DOES NOT SUPPORT PTRUE_{B/H?S/D} YET;
; RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%d --opcode-name=FADDV_VPZ_D 2>&1 | FileCheck %s --check-prefix=ZPR
; RUN: llvm-objdump -d %d > %t.s
; RUN: FileCheck %s --check-prefix=ZPR_ASM < %t.s
; ZPR-NOT: setRegTo is not implemented, results will be unreliable
; ZPR: assembled_snippet: {{.*}}C0035FD6
; ZPR_ASM: {{<foo>:}}
; ZPR_ASM: ptrue p{{[0-9]+}}.b
; ZPR_ASM-NEXT: dupm z{{[0-9]+}}.s, #0x1
; ZPR_ASM-NEXT: faddv d{{[0-9]+}}, p{{[0-9]+}}, z{{[0-9]+}}

# FPR128 REGISTER CLASS INITIALIZATION TESTCASE
; RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%d --opcode-name=ADDVv16i8v 2>&1 | FileCheck %s --check-prefix=FPR128
; RUN: llvm-objdump -d %d > %t.s
; RUN: FileCheck %s --check-prefix=FPR128-ASM < %t.s
; FPR128-NOT: setRegTo is not implemented, results will be unreliable
; FPR128: assembled_snippet: {{.*}}C0035FD6
; FPR128-ASM: {{<foo>:}}
; FPR128-ASM: movi v{{[0-9]+}}.2d, #0000000000000000
; FPR128-ASM-NEXT: addv b{{[0-9]+}}, v{{[0-9]+}}.16b

# FPR64 REGISTER CLASS INITIALIZATION TESTCASE
; RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%d --opcode-name=ADDVv4i16v 2>&1 | FileCheck %s --check-prefix=FPR64
; RUN: llvm-objdump -d %d > %t.s
; RUN: FileCheck %s --check-prefix=FPR64-ASM < %t.s
; FPR64-NOT: setRegTo is not implemented, results will be unreliable
; FPR64: assembled_snippet: {{.*}}C0035FD6
; FPR64-ASM: {{<foo>:}}
; FPR64-ASM: fmov d{{[0-9]+}}, {{#2.0+|#2\.000000000000000000e\+00}}
; FPR64-ASM-NEXT: addv h{{[0-9]+}}, v{{[0-9]+}}.4h
