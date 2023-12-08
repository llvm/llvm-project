# REQUIRES: aarch64-registered-target
# RUN: llvm-mc -triple=x86_64 %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s | llvm-readobj -hS - | FileCheck %s --check-prefixes=GNU,OBJ,OBJ-GNU
# RUN: llvm-mc -filetype=obj -triple=aarch64-freebsd %s | llvm-readobj -hS - | FileCheck %s --check-prefixes=FREEBSD,OBJ,OBJ-GNU
# RUN: llvm-mc -filetype=obj -triple=x86_64-solaris %s | llvm-readobj -hS - | FileCheck %s --check-prefixes=SOLARIS,OBJ,OBJ-SOLARIS

# ASM: .section retain,"aR",@progbits

## ELFOSABI_NONE is changed to ELFOSABI_GNU. Other OSABI values are unchanged.
# GNU:      OS/ABI: GNU/Linux
# FREEBSD:  OS/ABI: FreeBSD
# SOLARIS:  OS/ABI: Solaris

# OBJ:              Name: retain
# OBJ-NEXT:         Type: SHT_PROGBITS
# OBJ-NEXT:         Flags [
# OBJ-NEXT:           SHF_ALLOC
# OBJ-GNU-NEXT:       SHF_GNU_RETAIN
# OBJ-SOLARIS-NEXT:   SHF_SUNW_NODISCARD
# OBJ-NEXT:         ]

.section retain,"aR",@progbits
