# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -triple=x86_64 a.s | FileCheck %s

#--- a.s
# CHECK:      pushq %rax
# CHECK-NEXT: pushq %rbx
# CHECK-NEXT: pushq %rcx
.irp reg,%rax,%rbx
        pushq \reg
.endr
pushq %rcx

# CHECK:      addl %eax, 4
# CHECK-NEXT: addl %eax, 3
# CHECK-NEXT: addl %eax, 5
# CHECK-NEXT: addl %ebx, 4
# CHECK-NEXT: addl %ebx, 3
# CHECK-NEXT: addl %ebx, 5
# CHECK-EMPTY:
# CHECK-NEXT: nop
.irp reg,%eax,%ebx
.irp imm,4,3,5
        addl \reg, \imm
.endr # comment after .endr
.endr ;
nop

# CHECK:      xorl %eax, %eax
# CHECK-EMPTY:
# CHECK-NEXT: nop
.irp reg,%eax
xor \reg,\reg
.endr
# 99 "a.s"
nop

# RUN: not llvm-mc -triple=x86_64 err1.s 2>&1 | FileCheck %s --check-prefix=ERR1
# ERR1: .s:1:1: error: no matching '.endr' in definition
#--- err1.s
.irp reg,%eax
