# RUN: llvm-mc --filetype=obj --gsframe -triple x86_64 %s -o %t.o
# RUN: llvm-readelf --sframe %t.o | FileCheck %s

# Tests selection for the proper FRE::BX encoding at the boundaries
# between int8_t, int16_t, and int32_t.  Ensures the largest offset
# between CFA, RA, and FP governs. Align functions to 1024 to make it
# easier to interpet offsets. Some directives require alignment, so
# it isn't always possible to test exact boundaries.

# Also, check that irrelevant cfi directives don't create new fres,
# or affect the current ones. Checking the Start Address ensures that
# the proper FRE gets the proper checks. Using .long makes addresses
# architecture independent.

        .align 1024
fde4_fre_offset_sizes:
# CHECK:        FuncDescEntry [0] {
# CHECK:          Start FRE Offset: 0
# CHECK:            FRE Type: Addr1 (0x0)
        .cfi_startproc
# CHECK:        Frame Row Entry {
# CHECK-NEXT:          Start Address: 0x0
# CHECK-NEXT:          Return Address Signed: No
# CHECK-NEXT:          Offset Size: B1 (0x0)
# CHECK-NEXT:          Base Register: SP (0x1)
# CHECK-NEXT:          CFA Offset: 8
# CHECK-NEXT:          RA Offset: -8
        .long 0
# Uninteresting register no new fre, no effect on cfa
        .cfi_offset 0, 8
        .long 0
        .cfi_def_cfa_offset 0x78
# CHECK:        Frame Row Entry {
# CHECK-NEXT:          Start Address: 0x8
# CHECK-NEXT:          Return Address Signed: No
# CHECK-NEXT:          Offset Size: B1 (0x0)
# CHECK-NEXT:          Base Register: SP (0x1)
# CHECK-NEXT:          CFA Offset: 120
# CHECK-NEXT:          RA Offset: -8
        .long 0
# Uninteresting register no new fre, no effect on cfa
        .cfi_rel_offset 1, 8
        .long 0 
        .cfi_def_cfa_offset 0x80 
# CHECK:        Frame Row Entry {
# CHECK-NEXT:          Start Address: 0x10
# CHECK-NEXT:          Return Address Signed: No
# CHECK-NEXT:          Offset Size: B2 (0x1)
# CHECK-NEXT:          Base Register: SP (0x1)
# CHECK-NEXT:          CFA Offset: 128
# CHECK-NEXT:          RA Offset: -8
        .long 0
# Uninteresting register no new fre, no effect on cfa
        .cfi_val_offset 1, 8
        .long 0
        .cfi_def_cfa_offset 0x7FFF
# CHECK:        Frame Row Entry {
# CHECK-NEXT:          Start Address: 0x18
# CHECK-NEXT:          Return Address Signed: No
# CHECK-NEXT:          Offset Size: B2 (0x1)
# CHECK-NEXT:          Base Register: SP (0x1)
# CHECK-NEXT:          CFA Offset: 32767
# CHECK-NEXT:          RA Offset: -8
        .long 0
        .cfi_def_cfa_offset 0x8000
# CHECK:        Frame Row Entry {
# CHECK-NEXT:          Start Address: 0x1C
# CHECK-NEXT:          Return Address Signed: No
# CHECK-NEXT:          Offset Size: B4 (0x2)
# CHECK-NEXT:          Base Register: SP (0x1)
# CHECK-NEXT:          CFA Offset: 32768
# CHECK-NEXT:          RA Offset: -8
        .long 0
        .cfi_def_cfa_offset 0x8
# CHECK:        Frame Row Entry {
# CHECK-NEXT:          Start Address: 0x20
# CHECK-NEXT:          Return Address Signed: No
# CHECK-NEXT:          Offset Size: B1 (0x0)
# CHECK-NEXT:          Base Register: SP (0x1)
# CHECK-NEXT:          CFA Offset: 8
# CHECK-NEXT:          RA Offset: -8
        .long 0
        .cfi_adjust_cfa_offset 0x8
# CHECK:        Frame Row Entry {
# CHECK-NEXT:          Start Address: 0x24
# CHECK-NEXT:          Return Address Signed: No
# CHECK-NEXT:          Offset Size: B1 (0x0)
# CHECK-NEXT:          Base Register: SP (0x1)
# CHECK-NEXT:          CFA Offset: 16
# CHECK-NEXT:          RA Offset: -8
        .long 0
        .cfi_def_cfa_register  6  # switch to fp
# CHECK:        Frame Row Entry {
# CHECK-NEXT:          Start Address: 0x28
# CHECK-NEXT:          Return Address Signed: No
# CHECK-NEXT:          Offset Size: B1 (0x0)
# CHECK-NEXT:          Base Register: FP (0x0)
# CHECK-NEXT:          CFA Offset: 16
# CHECK-NEXT:          RA Offset: -8
        .long 0
        .cfi_offset 7, 32
        # sp not the cfa but with large offset still changes encoding.
        .cfi_offset 6, 0x7FF8
# CHECK:        Frame Row Entry {
# CHECK-NEXT:          Start Address: 0x2C
# CHECK-NEXT:          Return Address Signed: No
# CHECK-NEXT:          Offset Size: B2 (0x1)
# CHECK-NEXT:          Base Register: FP (0x0)
# CHECK-NEXT:          CFA Offset: 16
# CHECK-NEXT:          RA Offset: -8
# CHECK-NEXT:          FP Offset: 32760
        .long 0
        .cfi_endproc

        .align 1024
restore_reg:
# CHECK:        FuncDescEntry [1] {
# CHECK:          Start FRE Offset: 0x23
# CHECK-NEXT:         Num FREs: 3
        .cfi_startproc
# CHECK:        Frame Row Entry {
# CHECK-NEXT:        Start Address: 0x400
# CHECK-NOT        FP Offset{{.*}}
# CHECK:        }
        .long 0
        .cfi_offset 6, 32
# CHECK      Frame Row Entry {
# CHECK-NEXT     Start Address: 0x404
# CHECK:          FP Offset: 32
          .long 0
        .cfi_restore 6
# CHECK:        Frame Row Entry {
# CHECK-NEXT:        Start Address: 0x408
# CHECK-NOT        FP Offset{{.*}}
# CHECK:         }
        .long 0
        .cfi_endproc

        .align 1024
remember_restore_state:
# CHECK:        FuncDescEntry [2] {
# CHECK:          Start FRE Offset: 0x2D
# CHECK-NEXT:         Num FREs: 4
        .cfi_startproc
# CHECK:        Frame Row Entry {
# CHECK-NEXT:        Start Address: 0x800
# CHECK-NOT        FP Offset{{.*}}
# CHECK:        }
        .long 0
        .cfi_offset 6, 8
        .cfi_offset 7, 16
        .cfi_offset 8, 24
# CHECK:        Frame Row Entry {
# CHECK-NEXT:      Start Address: 0x804
# CHECK:          Base Register: SP (0x1)
# CHECK-NEXT:          CFA Offset: 8
# CHECK-NEXT:          RA Offset: -8
# CHECK-NEXT:          FP Offset: 8
# CHECK-NEXT:        }
        .long 0
        .cfi_remember_state
# CHECK:        Frame Row Entry {
# CHECK-NEXT:          Start Address: 0x808
# CHECK:          Base Register: SP (0x1)
# CHECK-NEXT:          CFA Offset: 8
# CHECK-NEXT:          RA Offset: -8
# CHECK-NEXT:          FP Offset: 32
# CHECK-NEXT:        }
        .cfi_offset 6, 32
        .cfi_offset 7, 40
        .cfi_offset 8, 48
        .long 0
# CHECK:        Frame Row Entry {
# CHECK-NEXT:      Start Address: 0x80C
# CHECK:          Base Register: SP (0x1)
# CHECK-NEXT:          CFA Offset: 8
# CHECK-NEXT:          RA Offset: -8
# CHECK-NEXT:          FP Offset: 8
# CHECK-NEXT:        }
        .cfi_restore_state
        .long 0

        .cfi_endproc
