# RUN: llvm-mc -triple=powerpc -filetype=obj %s -o %t.32.o
# RUN: llvm-objdump -d --no-show-raw-insn %t.32.o | FileCheck --check-prefixes=ELF32,CHECK %s

# RUN: llvm-mc -triple=powerpc64le -filetype=obj %s -o %t.64.o
# RUN: llvm-objdump -d --no-show-raw-insn %t.64.o | FileCheck --check-prefixes=ELF64,CHECK %s

# RUN: llvm-mc -triple=powerpc64 -filetype=obj %s -o %t.64.o
# RUN: llvm-objdump -d --no-show-raw-insn %t.64.o | FileCheck --check-prefixes=ELF64,CHECK %s

# CHECK-LABEL: <bl>:
# ELF32-NEXT:   bl .-4
# ELF64-NEXT:   bl .-4
# CHECK-NEXT:   bl .+0
# CHECK-NEXT:   bl .+4

bl:
  bl .-4
  bl .
  bl .+4

# CHECK-LABEL: <b>:
# CHECK-NEXT:   b .+67108860
# CHECK-NEXT:   b .+0
# CHECK-NEXT:   b .+4

b:
  b .-4
  b .
  b .+4

# CHECK-LABEL: <bt>:
# CHECK-NEXT:   bt 2, .+65532

bt:
  bt 2, .-4
