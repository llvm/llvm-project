// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t1 -split-dwarf-file %t2
// RUN: llvm-objdump -s %t1 | FileCheck --check-prefix=O %s
// RUN: llvm-objdump -s %t2 | FileCheck --check-prefix=DWO %s

// O-NOT: Contents of section
// O: Contents of section .strtab:
// O-NOT: Contents of section
// O: Contents of section .text:
// O-NEXT: 0000 c3
// O-NEXT: Contents of section .symtab:
// O-NOT: Contents of section
.globl main
main:
.Ltmp1:
ret
.Ltmp2:

// DWO-NOT: Contents of section
// DWO: Contents of section .strtab:
// DWO-NOT: Contents of section
// DWO: Contents of section .foo.dwo:
// DWO-NEXT: 0000 01000000
// DWO-NOT: Contents of section
.section .foo.dwo
.long .Ltmp2-.Ltmp1
