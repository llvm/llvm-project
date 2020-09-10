// RUN: llvm-mc -triple armv7a-linux-gnueabihf %s -filetype=obj -o /dev/null 2>&1 | FileCheck --check-prefix=OBJ --allow-empty %s
// RUN: not llvm-mc -triple armv7a-linux-gnueabihf %s -o /dev/null 2>&1 | FileCheck --check-prefix=ASM %s
// RUN: llvm-mc -triple armv7a-linux-gnueabihf %s -filetype=obj -o - | llvm-objdump -d - | FileCheck --check-prefix=DISASM %s

nop
// Create a new MCDataFragment due to Subtarget change
.arch_extension sec
9997:nop
.if . - 9997b == 0
// OBJ-NOT:[[@LINE-1]]:5: error: expected absolute expression
// ASM:[[@LINE-2]]:5: error: expected absolute expression
// DISASM: orr	r1, r1, #2
orr r1, r1, #1
.else
orr r1, r1, #2
.endif



@ RUN: not llvm-mc -filetype=obj -triple arm-linux-gnueabihf --defsym=ERR=1 %s -o /dev/null 2>&1 | FileCheck --check-prefix=ARM-ERR %s
@ RUN: not llvm-mc -filetype=obj -triple thumbv7a-linux-gnueabihf --defsym=ERR=1 %s -o /dev/null 2>&1 | FileCheck --check-prefix=THUMB2-ERR %s

.ifdef ERR
9997: nop
      .align 4
      nop
.if . - 9997b == 4
// ARM-ERR:[[@LINE-1]]:5: error: expected absolute expression
.endif

9997: nop
      .space 4
      nop
.if . - 9997b == 4
// ARM-ERR:[[@LINE-1]]:5: error: expected absolute expression
.endif

9997:
      ldr r0,=0x12345678
      .ltorg
      nop
.if . - 9997b == 4
// ARM-ERR:[[@LINE-1]]:5: error: expected absolute expression
.endif

9997: nop
      b external
      nop
.if . - 9997b == 4
// THUMB2-ERR:[[@LINE-1]]:5: error: expected absolute expression
.endif
.endif
