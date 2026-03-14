@ RUN: llvm-mc %s -triple armv5-unknown-linux -filetype=obj -o %t
@ RUN: llvm-objdump -d %t | FileCheck --check-prefix=STD %s
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

@ Test that the -M alias can be used flexibly. Create a baseline and ensure
@ all other combinations are identical.
@ RUN: llvm-objdump -d --disassembler-options=reg-names-raw %t > %t.raw
@ RUN: llvm-objdump -d -M reg-names-raw %t > %t.1
@ RUN: llvm-objdump -d -Mreg-names-raw %t > %t.2
@ RUN: llvm-objdump -d -Mreg-names-std -Mreg-names-raw %t > %t.3
@ RUN: llvm-objdump -d -Mreg-names-std,reg-names-raw %t > %t.4
@ RUN: llvm-objdump -dM reg-names-std,reg-names-raw %t > %t.5
@ RUN: llvm-objdump -dMreg-names-std,reg-names-raw %t > %t.6
@ RUN: llvm-objdump -dMreg-names-std -Mreg-names-raw %t > %t.7
@ RUN: cmp %t.raw %t.1
@ RUN: cmp %t.raw %t.2
@ RUN: cmp %t.raw %t.3
@ RUN: cmp %t.raw %t.4
@ RUN: cmp %t.raw %t.5
@ RUN: cmp %t.raw %t.6
@ RUN: cmp %t.raw %t.7

.text
  add r13, r14, r15
@ STD: add sp, lr, pc
@ RAW: add r13, r14, r15
