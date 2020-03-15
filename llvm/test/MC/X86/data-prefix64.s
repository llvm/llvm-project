# RUN: llvm-mc -triple=x86_64-unknown-unknown -filetype=obj %s -o - | llvm-objdump --triple=x86_64-unknown-unknown -d - | FileCheck %s

# CHECK: 66 0f 01 14 25 00 00 00 00
# CHECK: lgdtq 0
data16 lgdt 0

# CHECK: 66
# CHECK: data16
data16
