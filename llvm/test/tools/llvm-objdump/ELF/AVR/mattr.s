## When --mattr and --mcpu are both empty, disassemble all known instructions.
# RUN: llvm-mc -filetype=obj -triple=avr -mattr=+special %s -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s --check-prefixes=ALL

# ALL: <_start>:
# ALL-NEXT:      call    0x0
# ALL-NEXT:      jmp     0x0
# ALL-NEXT:      rjmp    .-2

## If --mattr or --mcpu is specified, don't default to --mattr=+special.
# RUN: llvm-objdump -d --no-show-raw-insn --mattr=+avr2 %t | FileCheck %s --check-prefixes=UNKNOWN
# RUN: llvm-objdump -d --no-show-raw-insn --mcpu=avr2 %t | FileCheck %s --check-prefixes=UNKNOWN

# UNKNOWN: <_start>:
# UNKNOWN-COUNT-2: <unknown>
# UNKNOWN:       rjmp    .-2

.globl _start
_start:
; Valid in avr3, Invalid in avr2
    call  foo
    jmp   foo
; Valid in both
    rjmp  foo
