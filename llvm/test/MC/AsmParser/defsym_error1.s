# RUN: not llvm-mc -filetype=obj -triple=i386-unknown-elf -defsym aaoeuaoeu %s 2>&1 | FileCheck %s
# CHECK: defsym must be of the form: sym=value
