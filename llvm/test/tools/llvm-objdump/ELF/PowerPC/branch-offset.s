# RUN: llvm-mc -triple=powerpc -filetype=obj %s -o %t.32be.o
# RUN: llvm-objdump -d --no-show-raw-insn %t.32be.o | FileCheck --check-prefixes=ELF32,CHECK %s

# RUN: llvm-mc -triple=powerpcle -filetype=obj %s -o %t.32le.o
# RUN: llvm-objdump -d --no-show-raw-insn %t.32le.o | FileCheck --check-prefixes=ELF32,CHECK %s

# RUN: llvm-mc -triple=powerpc64 -filetype=obj %s -o %t.64be.o
# RUN: llvm-objdump -d --no-show-raw-insn %t.64be.o | FileCheck --check-prefixes=ELF64,CHECK %s

# RUN: llvm-mc -triple=powerpc64le -filetype=obj %s -o %t.64le.o
# RUN: llvm-objdump -d --no-show-raw-insn %t.64le.o | FileCheck --check-prefixes=ELF64,CHECK %s

# CHECK-LABEL: <bl>:
# ELF32-NEXT:   bl 0xfffffffc
# ELF64-NEXT:   bl 0xfffffffffffffffc
# CHECK-NEXT:   bl 0x4
# CHECK-NEXT:   bl 0xc

bl:
  bl .-4
  bl .
  bl .+4

# CHECK-LABEL: <b>:
# CHECK-NEXT:   b 0x8
# CHECK-NEXT:   b 0x10
# CHECK-NEXT:   b 0x18

b:
  b .-4
  b .
  b .+4

# CHECK-LABEL: <bt>:
# CHECK-NEXT:   18: bt 2, 0x14
# CHECK-NEXT:   1c: bt 1, 0x20

bt:
  bt 2, .-4
  bgt .+4
