// RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s -o - | llvm-readobj -S --symbols --sd --addrsig | FileCheck %s

// CHECK:      Name: .llvm_addrsig
// CHECK-NEXT: VirtualSize: 0x0
// CHECK-NEXT: VirtualAddress: 0x0
// CHECK-NEXT: RawDataSize: 4
// CHECK-NEXT: PointerToRawData:
// CHECK-NEXT: PointerToRelocations: 0x0
// CHECK-NEXT: PointerToLineNumbers: 0x0
// CHECK-NEXT: RelocationCount: 0
// CHECK-NEXT: LineNumberCount: 0
// CHECK-NEXT: Characteristics [ (0x100800)
// CHECK-NEXT:   IMAGE_SCN_ALIGN_1BYTES (0x100000)
// CHECK-NEXT:   IMAGE_SCN_LNK_REMOVE (0x800)
// CHECK-NEXT: ]
// CHECK-NEXT: SectionData (
// CHECK-NEXT:   0000: 080A0B02
// CHECK-NEXT: )

// CHECK: Symbols [
// CHECK: Name: .text
// CHECK: AuxSectionDef
// CHECK: Name: .data
// CHECK: AuxSectionDef
// CHECK: Name: .bss
// CHECK: AuxSectionDef
// CHECK: Name: .llvm_addrsig
// CHECK: AuxSectionDef
// CHECK: Name: g1
// CHECK: Name: g2
// CHECK: Name: g3
// CHECK: Name: local

// CHECK:      Addrsig [
// CHECK-NEXT:   Sym: g1 (8)
// CHECK-NEXT:   Sym: g3 (10)
// CHECK-NEXT:   Sym: local (11)
// CHECK-NEXT:   Sym: .data (2)
// CHECK-NEXT: ]

.addrsig
.addrsig_sym g1
.globl g2
.addrsig_sym g3
.addrsig_sym local
.addrsig_sym .Llocal

local:

.data
.Llocal:
