# RUN: llvm-mc -triple=x86_64 %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-readelf -s %t | FileCheck %s

# ASM:      .symver nondef, nondef@v1, remove
# ASM-NEXT: nondef:
# ASM:      .symver def0, def0@@v2, remove
# ASM-NEXT: .symver def1, def1@@@v2{{$}}
# ASM-NEXT: def0:
# ASM-NEXT: def1:
# ASM:      .symver def2, def2@v1, remove
# ASM-NEXT: .symver def2, def2@@v2{{$}}
# ASM-NEXT: def2:

# CHECK:      1: 0000000000000000 0 NOTYPE GLOBAL DEFAULT [[#]] nondef@v1
# CHECK-NEXT: 2: 0000000000000000 0 NOTYPE GLOBAL DEFAULT [[#]] def0@@v2
# CHECK-NEXT: 3: 0000000000000000 0 NOTYPE GLOBAL DEFAULT [[#]] def1@@v2
# CHECK-NEXT: 4: 0000000000000000 0 NOTYPE GLOBAL DEFAULT [[#]] def2@v1
# CHECK-NEXT: 5: 0000000000000000 0 NOTYPE GLOBAL DEFAULT [[#]] def2@@v2
# CHECK-NOT:  {{.}}

.globl nondef
.symver nondef, nondef@v1, remove
nondef:

.globl def0, def1, def2
.symver def0, def0@@v2, remove
.symver def1, def1@@@v2, remove
def0:
def1:

## Giving multiple versions to the same original symbol is not useful.
## This test just documents the behavior.
.symver def2, def2@v1, remove
.symver def2, def2@@v2
def2:
