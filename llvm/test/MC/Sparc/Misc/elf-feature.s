## Emit correct SPARC v9 ELF flags depending on feature options.
## - `-mattr=+vis` sets the EF_SPARC_SUN_US1 flag; and
## - `-mattr=+vis2` sets the EF_SPARC_SUN_US3 flag.

# RUN: llvm-mc -filetype=obj -triple sparcv9              %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=COMMON      -DFLAG_VALUE=0x0                                %s
# RUN: llvm-mc -filetype=obj -triple sparcv9 -mattr=+vis  %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=COMMON,FLAG -DFLAG_VALUE=0x200 -DFLAG_NAME=EF_SPARC_SUN_US1 %s
# RUN: llvm-mc -filetype=obj -triple sparcv9 -mattr=+vis2 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=COMMON,FLAG -DFLAG_VALUE=0x800 -DFLAG_NAME=EF_SPARC_SUN_US3 %s

# COMMON:      Flags [ ([[FLAG_VALUE]])
# FLAG:          [[FLAG_NAME]]
# COMMON-NEXT: ]
