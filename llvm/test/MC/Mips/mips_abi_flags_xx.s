# RUN: llvm-mc %s -arch=mips -mcpu=mips32 | \
# RUN:   FileCheck %s -check-prefix=CHECK-ASM
#
# RUN: llvm-mc %s -arch=mips -mcpu=mips32 -filetype=obj -o - | \
# RUN:   llvm-readobj -sections -section-data -section-relocations -mips-abi-flags - | \
# RUN:   FileCheck %s -check-prefixes=CHECK-OBJ,CHECK-OBJ-32R1,CHECK-OBJ-MIPS

# RUN: llvm-mc /dev/null -arch=mips -mcpu=mips32 -mattr=fpxx -filetype=obj -o - | \
# RUN:   llvm-readobj -sections -section-data -section-relocations -mips-abi-flags - | \
# RUN:   FileCheck %s -check-prefixes=CHECK-OBJ,CHECK-OBJ-32R1,CHECK-OBJ-MIPS

# RUN: llvm-mc /dev/null -arch=mips -mcpu=mips32r6 -mattr=fpxx -filetype=obj -o - | \
# RUN:   llvm-readobj -sections -section-data -section-relocations -mips-abi-flags - | \
# RUN:   FileCheck %s -check-prefixes=CHECK-OBJ,CHECK-OBJ-32R6,CHECK-OBJ-MIPS

# RUN: llvm-mc /dev/null -arch=mips -mcpu=octeon -target-abi n64 -filetype=obj -o - | \
# RUN:   llvm-readobj -sections -section-data -section-relocations -mips-abi-flags - | \
# RUN:   FileCheck %s -check-prefixes=CHECK-OBJ,CHECK-OBJ-64R2,CHECK-OBJ-OCTEON

# CHECK-ASM: .module fp=xx

# Checking if the Mips.abiflags were correctly emitted.
# CHECK-OBJ:       Section {
# CHECK-OBJ:         Index: 5
# CHECK-OBJ-LABEL:   Name: .MIPS.abiflags
# CHECK-OBJ:         Type: SHT_MIPS_ABIFLAGS (0x7000002A)
# CHECK-OBJ:          Flags [ (0x2)
# CHECK-OBJ:           SHF_ALLOC (0x2)
# CHECK-OBJ:         ]
# CHECK-OBJ:         Address: 0x0
# CHECK-OBJ:         Size: 24
# CHECK-OBJ:         Link: 0
# CHECK-OBJ:         Info: 0
# CHECK-OBJ:         AddressAlignment: 8
# CHECK-OBJ:         EntrySize: 24
# CHECK-OBJ:         Relocations [
# CHECK-OBJ:         ]
# CHECK-OBJ-LABEL: }
# CHECK-OBJ:       MIPS ABI Flags {
# CHECK-OBJ-NEXT:    Version: 0
# CHECK-OBJ-32R1-NEXT: ISA: {{MIPS32$}}
# CHECK-OBJ-32R6-NEXT: ISA: MIPS32r6
# CHECK-OBJ-64R2-NEXT: ISA: MIPS64r2
# CHECK-OBJ-MIPS-NEXT:   ISA Extension: None (0x0)
# CHECK-OBJ-OCTEON-NEXT: ISA Extension: Cavium Networks Octeon (0x5)
# CHECK-OBJ-NEXT:    ASEs [ (0x0)          
# CHECK-OBJ-NEXT:    ]                     
# CHECK-OBJ-32R1-NEXT: FP ABI: Hard float (32-bit CPU, Any FPU) (0x5)
# CHECK-OBJ-32R6-NEXT: FP ABI: Hard float (32-bit CPU, Any FPU) (0x5)
# CHECK-OBJ-64R2-NEXT: FP ABI: Hard float (double precision) (0x1)
# CHECK-OBJ-32R1-NEXT: GPR size: 32
# CHECK-OBJ-32R6-NEXT: GPR size: 32
# CHECK-OBJ-64R2-NEXT: GPR size: 64
# CHECK-OBJ-32R1-NEXT: CPR1 size: 32
# CHECK-OBJ-32R6-NEXT: CPR1 size: 32
# CHECK-OBJ-64R2-NEXT: CPR1 size: 64
# CHECK-OBJ-NEXT:    CPR2 size: 0
# CHECK-OBJ-NEXT:    Flags 1 [ (0x1)
# CHECK-OBJ-NEXT:      ODDSPREG (0x1)
# CHECK-OBJ-NEXT:    ]
# CHECK-OBJ-NEXT:    Flags 2: 0x0
# CHECK-OBJ-NEXT:  }

        .module fp=xx

# FIXME: Test should include gnu_attributes directive when implemented.
#        An explicit .gnu_attribute must be checked against the effective
#        command line options and any inconsistencies reported via a warning.
