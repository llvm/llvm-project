# RUN: llvm-mc -filetype=obj -arch=x86 -triple=x86_64-pc-linux-gnu -mcpu=pentiumpro %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefix=LNOP10
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu %s -mcpu=pentiumpro | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefix=LNOP10
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=x86_64-apple-darwin10.0 -mcpu=pentiumpro %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefix=LNOP10
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-apple-darwin8 -mcpu=pentiumpro %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefix=LNOP10
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu -mcpu=slm %s | llvm-objdump -d --no-show-raw-insn - | FileCheck --check-prefix=LNOP7 %s
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu -mcpu=silvermont %s | llvm-objdump -d --no-show-raw-insn - | FileCheck --check-prefix=LNOP7 %s
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu -mcpu=lakemont %s | llvm-objdump -d --no-show-raw-insn - | FileCheck --check-prefix=NOP1 %s
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=x86_64-pc-linux-gnu -mcpu=bdver1 %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefix=LNOP11
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu %s -mcpu=bdver1 | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefix=LNOP11
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=x86_64-pc-linux-gnu -mcpu=btver1 %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefix=LNOP15
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu %s -mcpu=btver1 | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefix=LNOP15
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=x86_64-pc-linux-gnu -mcpu=btver2 %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefix=LNOP15
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu %s -mcpu=btver2 | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefix=LNOP15
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=x86_64-pc-linux-gnu -mcpu=znver1 %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefix=LNOP15
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu %s -mcpu=znver1 | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefix=LNOP15
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=x86_64-pc-linux-gnu -mcpu=znver2 %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefix=LNOP15
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu %s -mcpu=znver2 | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefix=LNOP15
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=x86_64-pc-linux-gnu -mcpu=znver3 %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefix=LNOP15
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu %s -mcpu=znver3 | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefix=LNOP15
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=x86_64-pc-linux-gnu -mcpu=znver4 %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefix=LNOP15
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu %s -mcpu=znver4 | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefix=LNOP15
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu -mcpu=nehalem %s | llvm-objdump -d --no-show-raw-insn - | FileCheck --check-prefix=LNOP10 %s
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu -mcpu=westmere %s | llvm-objdump -d --no-show-raw-insn - | FileCheck --check-prefix=LNOP10 %s
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu -mcpu=sandybridge %s | llvm-objdump -d --no-show-raw-insn - | FileCheck --check-prefix=LNOP15 %s
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu -mcpu=ivybridge %s | llvm-objdump -d --no-show-raw-insn - | FileCheck --check-prefix=LNOP15 %s
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu -mcpu=haswell %s | llvm-objdump -d --no-show-raw-insn - | FileCheck --check-prefix=LNOP15 %s
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu -mcpu=broadwell %s | llvm-objdump -d --no-show-raw-insn - | FileCheck --check-prefix=LNOP15 %s
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu -mcpu=skylake %s | llvm-objdump -d --no-show-raw-insn - | FileCheck --check-prefix=LNOP15 %s
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu -mcpu=skx %s | llvm-objdump -d --no-show-raw-insn - | FileCheck --check-prefix=LNOP15 %s
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu -mcpu=knl %s | llvm-objdump -d --no-show-raw-insn - | FileCheck --check-prefix=LNOP10 %s
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu -mcpu=knm %s | llvm-objdump -d --no-show-raw-insn - | FileCheck --check-prefix=LNOP10 %s

# Ensure alignment directives also emit sequences of 10, 11 and 15-byte NOPs on processors
# capable of using long NOPs.
inc %eax
.p2align 5
inc %eax
# LNOP15: 0:  inc
# LNOP15-NEXT: 1:  nop
# LNOP15-NEXT: 10: nop
# LNOP15-NEXT: 1f: nop
# LNOP15-NEXT: 20: inc

# LNOP11: 0:  inc
# LNOP11-NEXT: 1:  nop
# LNOP11-NEXT: c:  nop
# LNOP11-NEXT: 17: nop
# LNOP11-NEXT: 20: inc

# LNOP10: 0:  inc
# LNOP10-NEXT: 1:  nop
# LNOP10-NEXT: b:  nop
# LNOP10-NEXT: 15: nop
# LNOP10-NEXT: 1f: nop
# LNOP10-NEXT: 20: inc

# On Silvermont we emit only 7 byte NOPs since longer NOPs are not profitable.
# LNOP7: 0:  inc
# LNOP7-NEXT: 1:  nop
# LNOP7-NEXT: 8:  nop
# LNOP7-NEXT: f:  nop
# LNOP7-NEXT: 16: nop
# LNOP7-NEXT: 1d: nop
# LNOP7-NEXT: 20: inc

# On Lakemont we emit only 1 byte NOPs since longer NOPs are not supported/legal
# NOP1: 0:  inc
# NOP1-NEXT: 1:  nop
# NOP1-NEXT: 2:  nop
# NOP1-NEXT: 3:  nop
# NOP1-NEXT: 4:  nop
# NOP1-NEXT: 5:  nop
# NOP1-NEXT: 6:  nop
# .......
# NOP1: 17:  nop
# NOP1-NEXT: 18:  nop
# NOP1-NEXT: 19:  nop
# NOP1-NEXT: 1a:  nop
# NOP1-NEXT: 1b:  nop
# NOP1-NEXT: 1c:  nop
# NOP1-NEXT: 1d:  nop
# NOP1-NEXT: 1e:  nop
# NOP1-NEXT: 1f:  nop
# NOP1-NEXT: 20:  inc
