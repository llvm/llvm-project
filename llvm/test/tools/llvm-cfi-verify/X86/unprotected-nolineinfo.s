# RUN: llvm-mc %S/Inputs/unprotected-nolineinfo.s -filetype obj \
# RUN:         -triple x86_64-linux-elf -o %t.o
# RUN: not llvm-cfi-verify %t.o 2>&1 | FileCheck %s

# CHECK: DWARF line information missing. Did you compile with '-g'?
