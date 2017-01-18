# RUN: llvm-mc -triple i386-unknown-unknown-code16 -filetype=obj %s -o - | llvm-objdump -triple i386-unknown-unknown-code16 -d - | FileCheck %s

# CHECK: 66 0f 01 16 00 00
# CHECK: lgdtl 0
data32 lgdt 0

# CHECK: 66
# CHECK: data32
data32
