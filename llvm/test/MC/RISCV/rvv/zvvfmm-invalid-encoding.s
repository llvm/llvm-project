# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvvfmm %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-zvvfmm - \
# RUN:        | FileCheck %s

# vm=0 is reserved for non-widening vfmmacc.vv, so this raw encoding must not
# decode as a scaled vfmmacc form.
.insn 0x4, 0x51421457
# CHECK: <unknown>
# CHECK-NOT: vfmmacc.vv
