# RUN: llvm-mc -triple=x86_64 %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s | llvm-readobj -hS - | FileCheck %s --check-prefix=OBJ

# ASM: .section retain,"aR",@progbits

## Note: GNU as sets OSABI to GNU.
# OBJ:      OS/ABI: SystemV (0x0)

# OBJ:      Name: retain
# OBJ-NEXT: Type: SHT_PROGBITS
# OBJ-NEXT: Flags [
# OBJ-NEXT:   SHF_ALLOC
# OBJ-NEXT:   SHF_GNU_RETAIN
# OBJ-NEXT: ]

.section retain,"aR",@progbits
