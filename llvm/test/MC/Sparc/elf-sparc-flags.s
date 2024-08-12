## Emit correct flags depending on triple, cpu, and memory model options.
## - `-triple sparc` sets the flag field to 0x0
## - `-triple sparc -mattr=+v8plus` adds an EF_SPARC_32PLUS (0x100)
## - Currently, for sparc64 we always compile for TSO memory model, so
##   `-triple sparcv9` sets the memory model flag to EF_SPARCV9_TSO (0x0)
##   (i.e the last two bits have to be a zero).

# RUN: llvm-mc -filetype=obj -triple sparc                  %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=SPARC       %s
# RUN: llvm-mc -filetype=obj -triple sparc   -mattr=+v8plus %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=SPARC32PLUS %s
# RUN: llvm-mc -filetype=obj -triple sparcv9                %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=SPARCV9     %s

# SPARC:      Flags [ (0x0)
# SPARC-NEXT: ]

# SPARC32PLUS:      Flags [ (0x100)
# SPARC32PLUS-NEXT:   EF_SPARC_32PLUS (0x100)
# SPARC32PLUS-NEXT: ]

# SPARCV9:      Flags [ (0x0)
# SPARCV9-NEXT: ]
