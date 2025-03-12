// RUN: llvm-mc -triple aarch64-windows-msvc -filetype obj -o %t.obj %s
// RUN: llvm-readobj --sections --sd --relocs %t.obj | FileCheck %s

.section        nc_sect,"xr"
normal_call:
  str     x30, [sp, #-16]!                // 8-byte Folded Spill
  adrp    x8, __imp_a
  ldr     x8, [x8, :lo12:__imp_a]
.Limpcall0:
  blr     x8
  ldr     x30, [sp], #16                  // 8-byte Folded Reload
  ret

.section        tc_sect,"xr"
tail_call:
  adrp    x8, __imp_b
  ldr     x8, [x8, :lo12:__imp_b]
.Limpcall1:
  br     x8

.section        .impcall,"yi"
.asciz  "Imp_Call_V1"
.word   20
.secnum nc_sect
.word   19
.secoffset      .Limpcall0
.symidx __imp_a
.word   20
.secnum tc_sect
.word   19
.secoffset      .Limpcall1
.symidx __imp_b

// CHECK-LABEL: Name: .impcall (2E 69 6D 70 63 61 6C 6C)
// CHECK-NEXT:  VirtualSize: 0x0
// CHECK-NEXT:  VirtualAddress: 0x0
// CHECK-NEXT:  RawDataSize: 52
// CHECK-NEXT:  PointerToRawData: 0x150
// CHECK-NEXT:  PointerToRelocations: 0x0
// CHECK-NEXT:  PointerToLineNumbers: 0x0
// CHECK-NEXT:  RelocationCount: 0
// CHECK-NEXT:  LineNumberCount: 0
// CHECK-NEXT:  Characteristics [
// CHECK-NEXT:    IMAGE_SCN_ALIGN_4BYTES
// CHECK-NEXT:    IMAGE_SCN_LNK_INFO
// CHECK-NEXT:  ]
// CHECK-NEXT:  SectionData (
// CHECK-NEXT:    0000: 496D705F 43616C6C 5F563100 14000000  |Imp_Call_V1.....|
// CHECK-NEXT:    0010:
// CHECK-SAME:    [[#%.2X,NCSECT:]]000000
// CHECK-SAME:    13000000
// CHECK-SAME:    [[#%.2X,NCOFFSET:]]000000
// CHECK-SAME:    [[#%.2X,NCSYM:]]000000
// CHECK-NEXT:    0020:
// CHECK-SAME:    14000000
// CHECK-SAME:    [[#%.2X,TCSECT:]]000000
// CHECK-SAME:    13000000
// CHECK-SAME:    [[#%.2X,TCOFFSET:]]000000
// CHECK-NEXT:    0030:
// CHECK-SAME:    [[#%.2X,TCSYM:]]000000
// CHECK-NEXT:  )

// CHECK-LABEL: Relocations [
// CHECK-NEXT:     Section ([[#%u,NCSECT]]) nc_sect {
// CHECK-NEXT:       0x[[#%x,NCOFFSET - 8]] IMAGE_REL_ARM64_PAGEBASE_REL21 __imp_a ([[#%u,NCSYM]])
// CHECK-NEXT:       0x[[#%x,NCOFFSET - 4]] IMAGE_REL_ARM64_PAGEOFFSET_12L __imp_a ([[#%u,NCSYM]])
// CHECK-NEXT:     }
// CHECK-NEXT:     Section ([[#%u,TCSECT]]) tc_sect {
// CHECK-NEXT:       0x[[#%x,TCOFFSET - 8]] IMAGE_REL_ARM64_PAGEBASE_REL21 __imp_b ([[#%u,TCSYM]])
// CHECK-NEXT:       0x[[#%x,TCOFFSET - 4]] IMAGE_REL_ARM64_PAGEOFFSET_12L __imp_b ([[#%u,TCSYM]])
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
