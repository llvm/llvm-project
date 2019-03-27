@ RUN: llvm-mc %s -triple armv5-unknown-linux -filetype=obj -o %t
@ RUN: llvm-objdump -d %t | FileCheck -check-prefix=STD %s
@ RUN: llvm-objdump -d -Mreg-names-std %t \
@ RUN:   | FileCheck -check-prefix=STD %s
@ RUN: llvm-objdump -d --disassembler-options=reg-names-raw %t \
@ RUN:   | FileCheck -check-prefix=RAW %s
@ RUN: llvm-objdump -d -Mreg-names-raw,reg-names-std %t \
@ RUN:   | FileCheck -check-prefix=STD %s
@ RUN: llvm-objdump -d -Mreg-names-std,reg-names-raw %t \
@ RUN:   | FileCheck -check-prefix=RAW %s
@ RUN: not llvm-objdump -d -Munknown %t 2>&1 \
@ RUN:   | FileCheck -check-prefix=ERR %s
@ ERR: Unrecognized disassembler option: unknown

.text
  add r13, r14, r15
@ STD: add sp, lr, pc
@ RAW: add r13, r14, r15
