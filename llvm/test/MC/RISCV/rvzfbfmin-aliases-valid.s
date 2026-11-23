# RUN: llvm-mc %s -triple=riscv32 -mattr=+zfbfmin -M no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+zfbfmin \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zfbfmin -M no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zfbfmin \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zfbfmin < %s \
# RUN:     | llvm-objdump -d --mattr=+zfbfmin --no-print-imm-hex -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zfbfmin < %s \
# RUN:     | llvm-objdump -d --mattr=+zfbfmin --no-print-imm-hex - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+zfbfmin < %s \
# RUN:     | llvm-objdump -d --mattr=+zfbfmin --no-print-imm-hex -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+zfbfmin < %s \
# RUN:     | llvm-objdump -d --mattr=+zfbfmin --no-print-imm-hex - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s

##===----------------------------------------------------------------------===##
## Assembler Pseudo Instructions (User-Level ISA, Version 2.2, Chapter 20)
##===----------------------------------------------------------------------===##

# CHECK-INST: flh ft0, 0(a0)
# CHECK-ALIAS: flh ft0, 0(a0)
flh f0, (x10)
# CHECK-INST: fsh ft0, 0(a0)
# CHECK-ALIAS: fsh ft0, 0(a0)
fsh f0, (x10)
