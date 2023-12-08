# RUN: llvm-mc %s -triple mips-unknown-linux-gnu | \
# RUN:   FileCheck %s -check-prefix=CHECK-ASM
#
# RUN: llvm-mc %s -triple mips-unknown-linux-gnu -filetype=obj -o - | \
# RUN:   llvm-readobj --sections --section-data --section-relocations -A - | \
# RUN:     FileCheck %s -check-prefix=CHECK-OBJ

# CHECK-ASM: .module fp=xx
# CHECK-ASM: .set    fp=64

# Checking if the Mips.abiflags were correctly emitted.
# CHECK-OBJ:       Section {
# CHECK-OBJ:         Index: 5
# CHECK-OBJ-LABEL:   Name: .MIPS.abiflags (12)
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
# CHECK-OBJ-NEXT:    ISA: {{MIPS32$}}
# CHECK-OBJ-NEXT:    ISA Extension: None (0x0)
# CHECK-OBJ-NEXT:    ASEs [ (0x0)
# CHECK-OBJ-NEXT:    ]
# CHECK-OBJ-NEXT:    FP ABI: Hard float (32-bit CPU, Any FPU) (0x5)
# CHECK-OBJ-NEXT:    GPR size: 32
# CHECK-OBJ-NEXT:    CPR1 size: 32
# CHECK-OBJ-NEXT:    CPR2 size: 0
# CHECK-OBJ-NEXT:    Flags 1 [ (0x1)
# CHECK-OBJ-NEXT:      ODDSPREG (0x1)
# CHECK-OBJ-NEXT:    ]
# CHECK-OBJ-NEXT:    Flags 2: 0x0
# CHECK-OBJ-NEXT:  }

        .module fp=xx
        .set    fp=64
# FIXME: Test should include gnu_attributes directive when implemented.
#        An explicit .gnu_attribute must be checked against the effective
#        command line options and any inconsistencies reported via a warning.
