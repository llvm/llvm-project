# RUN: not llvm-mc -filetype=obj -triple=i386-unknown-elf -defsym a=a %s 2>&1 | FileCheck %s
# CHECK: error: value is not an integer: a
