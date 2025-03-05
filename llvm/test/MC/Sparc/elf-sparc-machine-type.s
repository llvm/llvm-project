## Emit correct machine type depending on triple and cpu options.
## - `-triple sparc` emits an object of type EM_SPARC;
## - `-triple sparc -mattr=+v8plus` emits EM_SPARC32PLUS; and
## - `-triple sparcv9` emits EM_SPARCV9.

# RUN: llvm-mc -filetype=obj -triple sparc                  %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=SPARC       %s
# RUN: llvm-mc -filetype=obj -triple sparc   -mattr=+v8plus %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=SPARC32PLUS %s
# RUN: llvm-mc -filetype=obj -triple sparcv9                %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=SPARCV9     %s

# SPARC:       Machine: EM_SPARC (0x2)
# SPARC32PLUS: Machine: EM_SPARC32PLUS (0x12)
# SPARCV9:     Machine: EM_SPARCV9 (0x2B)
