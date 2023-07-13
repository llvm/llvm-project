## Test that we default to --mcpu=future and disassemble all known instructions.
## The default is different from producers (e.g. Clang).
# RUN: llvm-mc -triple=powerpc64le -filetype=obj %s -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s --check-prefixes=CHECK,FUTURE
# RUN: llvm-objdump -d --no-show-raw-insn --mcpu=future %t | FileCheck %s --check-prefixes=CHECK,FUTURE
# RUN: llvm-objdump -d --no-show-raw-insn --mcpu=pwr9 %t | FileCheck %s --check-prefixes=CHECK,UNKNOWN

# RUN: llvm-mc -triple=powerpc -filetype=obj %s -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s --check-prefixes=CHECK,FUTURE

# CHECK-LABEL: <_start>:
# FUTURE-NEXT:   pld 3, 0(0), 1
# UNKNOWN-COUNT-2: <unknown>
# CHECK-EMPTY:

.globl _start
_start:
  pld 3, 0(0), 1
