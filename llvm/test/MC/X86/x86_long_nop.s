# RUN: llvm-mc -filetype=obj -arch=x86 -triple=x86_64-pc-linux-gnu %s | llvm-objdump -d -no-show-raw-insn - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu %s | llvm-objdump -d -no-show-raw-insn - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=x86_64-apple-darwin10.0 %s | llvm-objdump -d -no-show-raw-insn - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-apple-darwin8 %s | llvm-objdump -d -no-show-raw-insn - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu -mcpu=slm %s | llvm-objdump -d -no-show-raw-insn - | FileCheck --check-prefix=LNOP7 %s
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu -mcpu=lakemont %s | llvm-objdump -d -no-show-raw-insn - | FileCheck --check-prefix=LNOP7 %s

# Ensure alignment directives also emit sequences of 15-byte NOPs on processors
# capable of using long NOPs.
inc %eax
.p2align 5
inc %eax
# CHECK: 0:  inc
# CHECK-NEXT: 1:  nop
# CHECK-NEXT: 10:  nop
# CHECK-NEXT: 1f:  nop
# CHECK-NEXT: 20:  inc

# On Silvermont and Lakemont we emit only 7 byte NOPs since longer NOPs
# are not profitable.
# LNOP7: 0:  inc
# LNOP7-NEXT: 1:  nop
# LNOP7-NEXT: 8:  nop
# LNOP7-NEXT: f:  nop
# LNOP7-NEXT: 16:  nop
# LNOP7-NEXT: 1d:  nop
# LNOP7-NEXT: 20:  inc
