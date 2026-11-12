# RUN: llvm-mc -triple=i386-unknown-unknown -filetype=obj %s -o - | llvm-objdump --triple=i386-unknown-unknown -d - | FileCheck %s

# CHECK: 66 0f 01 15 00 00 00 00
# CHECK: lgdtw 0
data16 lgdt 0

# CHECK: 66
# CHECK: data16
data16
