# RUN: llvm-mc -filetype=obj -triple=mips64 %s -o %t.be
# RUN: llvm-objdump -d %t.be | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=mips64el %s -o %t.le
# RUN: llvm-objdump -d %t.le | FileCheck %s

# RUN: not llvm-mc -filetype=obj -triple=mips64el --defsym ERR=1 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

# CHECK:      addiu $4, $5, -0x8000
# CHECK-NEXT: addiu $4, $5, -0x1
# CHECK-NEXT: addiu $4, $5, -0x8000
# CHECK-NEXT: addiu $4, $5, 0x7fff
# CHECK-NEXT: addiu $4, $5, -0x1
addiu $4, $5, v_32769+1
addiu $4, $5, v65535
addiu $4, $5, .L0-.L1
addiu $4, $5, .L2-.L1
addiu $4, $5, .L2-.L0+0

# CHECK:      andi $4, $5, 0xffff
# CHECK:      slti $4, $5, -0x1
andi $4, $5, v65535 # uimm16
slti $4, $5, v65535 # simm16

.ifdef ERR
# ERR: :[[#@LINE+1]]:15: error: fixup value out of range [-32768, 65535]
addiu $4, $5, v_32769
# ERR: :[[#@LINE+1]]:21: error: fixup value out of range [-32768, 65535]
addiu $4, $5, v65535+1

# ERR: [[#@LINE+1]]:18: error: fixup value out of range [-32768, 65535]
addiu $4, $5, .L2-.L0+1
.endif

v_32769 = -32769
v65535 = 65535

.section .rodata,"a"
.L0:
.space 32768
.L1:
.space 32767
.L2:
