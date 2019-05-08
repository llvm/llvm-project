# RUN: yaml2obj %p/Inputs/simple-executable-x86_64.yaml -o %t
# RUN: llvm-objdump -d %t | FileCheck %s

# CHECK:       :      file format ELF64-x86-64
# CHECK-EMPTY:
# CHECK-EMPTY:
# CHECK-NEXT:  Disassembly of section .text:
# CHECK-EMPTY:
# CHECK-NEXT:  0000000000000000 foo:
