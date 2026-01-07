## When --mattr and --mcpu are both empty, disassemble all known instructions.
# RUN: llvm-mc -filetype=obj -triple=avr -mattr=+special %s -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s --check-prefixes=CHECK,ALL

## If --mattr or --mcpu is specified, don't default to --mattr=+special.
# RUN: llvm-objdump -d --no-show-raw-insn --mattr=+avr2 %t | FileCheck %s --check-prefixes=CHECK,UNKNOWN
# RUN: llvm-objdump -d --no-show-raw-insn --mcpu=avr2 %t | FileCheck %s --check-prefixes=CHECK,UNKNOWN

# CHECK-LABEL: <_start>:
# ALL-NEXT:      call    0x0
# ALL-NEXT:      jmp     0x0
# ALL-NEXT:      rjmp    .-2
# UNKNOWN-COUNT-2: <unknown>
# UNKNOWN:       rjmp    .-2

.globl _start
_start:
; Valid in avr3, Invalid in avr2
    call  dummp_symbol
    jmp   dummp_symbol
; Valid in both
    rjmp  dummp_symbol
