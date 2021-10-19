# RUN: not llvm-mc -triple hexagon < %s 2>&1 | FileCheck %s

# CHECK: :[[#@LINE+1]]:7: error: out of range literal value
.byte 0xffa
# CHECK: :[[#@LINE+1]]:7: error: out of range literal value
.half 0xffffa
# CHECK: :[[#@LINE+1]]:8: error: out of range literal value
.short 0xffffa
# CHECK: :[[#@LINE+1]]:8: error: out of range literal value
.hword 0xffffa
# CHECK: :[[#@LINE+1]]:8: error: out of range literal value
.2byte 0xffffa
# CHECK: :[[#@LINE+1]]:7: error: out of range literal value
.word 0xffffffffa
# CHECK: :[[#@LINE+1]]:7: error: out of range literal value
.long 0xffffffffa
# CHECK: :[[#@LINE+1]]:8: error: out of range literal value
.4byte 0xffffffffa
# CHECK: :[[#@LINE+1]]:8: error: literal value out of range for directive
.8byte 0xffffffffffffffffa
