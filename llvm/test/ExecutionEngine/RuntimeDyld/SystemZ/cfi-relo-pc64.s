// Test that R_390_PC32 and R_390_PC64 relocations can be generated.
// RUN: llvm-mc -triple s390x-linux-gnu -filetype=obj %s -o - | llvm-readobj -S --sr --sd - | FileCheck %s

// Test that RuntimeDyld can fix up such relocations.
// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-mc -triple s390x-linux-gnu  -filetype=obj %s -o %t/test-s390x-cfi-relo-pc64.o
// RUN: llc -mtriple=s390x-linux-gnu -filetype=obj %S/Inputs/rtdyld-globals.ll -o %t/test-s390x-rtdyld-globals.o
// RUN: llvm-rtdyld -triple=s390x-linux-gnu -verify %t/test-s390x-cfi-relo-pc64.o %t/test-s390x-rtdyld-globals.o

f1:
    .cfi_startproc
    .cfi_personality 0x9c, foo // DW_EH_PE_indirect|DW_EH_PE_pcrel|DW_EH_PE_sdata8 (0x9c)
    lr %r0, %r0
    .cfi_endproc

// CHECK:        Section {
// CHECK:          Index:
// CHECK:          Name: .rela.eh_frame
// CHECK-NEXT:     Type: SHT_RELA
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_INFO_LINK
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 48
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment: 8
// CHECK-NEXT:     EntrySize: 24
// CHECK-NEXT:     Relocations [
// CHECK-NEXT:       0x12 R_390_PC64 foo 0x0
// CHECK-NEXT:       0x28 R_390_PC32 .text 0x0
// CHECK-NEXT:     ]
// CHECK:        }
