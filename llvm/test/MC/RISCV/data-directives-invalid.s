# RUN: not llvm-mc -triple riscv32 < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 < %s 2>&1 | FileCheck %s

# CHECK: [[@LINE+1]]:7: error: out of range literal value in '.byte' directive
.byte 0xffa
# CHECK: [[@LINE+1]]:7: error: out of range literal value in '.half' directive
.half 0xffffa
# CHECK: [[@LINE+1]]:8: error: out of range literal value in '.short' directive
.short 0xffffa
# CHECK: [[@LINE+1]]:8: error: out of range literal value in '.hword' directive
.hword 0xffffa
# CHECK: [[@LINE+1]]:8: error: out of range literal value in '.2byte' directive
.2byte 0xffffa
# CHECK: [[@LINE+1]]:7: error: out of range literal value in '.word' directive
.word 0xffffffffa
# CHECK: [[@LINE+1]]:7: error: out of range literal value in '.long' directive
.long 0xffffffffa
# CHECK: [[@LINE+1]]:8: error: out of range literal value in '.4byte' directive
.4byte 0xffffffffa
# CHECK: [[@LINE+1]]:8: error: literal value out of range for directive in '.dword' directive
.dword 0xffffffffffffffffa
# CHECK: [[@LINE+1]]:8: error: literal value out of range for directive in '.8byte' directive
.8byte 0xffffffffffffffffa
