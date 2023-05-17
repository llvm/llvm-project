## When --mattr and --mcpu are both empty, disassemble all known instructions.
# RUN: llvm-mc -filetype=obj -triple=aarch64 -mattr=+all %s -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s --check-prefixes=CHECK,ALL

## If --mattr or --mcpu is specified, don't default to --mattr=+all.
# RUN: llvm-objdump -d --no-show-raw-insn --mattr=+v8a %t | FileCheck %s --check-prefixes=CHECK,UNKNOWN
# RUN: llvm-objdump -d --no-show-raw-insn --mcpu=generic %t | FileCheck %s --check-prefixes=CHECK,UNKNOWN

# CHECK-LABEL: <_start>:
# ALL-NEXT:      bc.eq 0x4
# ALL-NEXT:      irg x0, x1
# ALL-NEXT:      mrs x0, RNDR
# UNKNOWN-COUNT-2: <unknown>
# UNKNOWN:       mrs x0, S3_3_C2_C4_0
# CHECK-EMPTY:

.globl _start
_start:
  bc.eq #4      // armv8.8-a hbc
  irg x0, x1    // armv8.5-a mte
  mrs x0, RNDR  // armv8.5-a rand
