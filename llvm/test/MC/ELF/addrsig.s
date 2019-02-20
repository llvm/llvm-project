// RUN: llvm-mc -filetype=asm -triple x86_64-pc-linux-gnu %s -o - | FileCheck --check-prefix=ASM %s
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -s -t -sd -addrsig | FileCheck %s
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -split-dwarf-file %t.dwo -o - | llvm-readobj -s -t -sd -addrsig | FileCheck %s
// RUN: llvm-readobj -s %t.dwo | FileCheck --check-prefix=DWO %s

// CHECK:        Name: .llvm_addrsig
// CHECK-NEXT:   Type: SHT_LLVM_ADDRSIG (0x6FFF4C03)
// CHECK-NEXT:   Flags [ (0x80000000)
// CHECK-NEXT:     SHF_EXCLUDE (0x80000000)
// CHECK-NEXT:   ]
// CHECK-NEXT:   Address:
// CHECK-NEXT:   Offset:
// CHECK-NEXT:   Size: 4
// CHECK-NEXT:   Link: 4
// CHECK-NEXT:   Info: 0
// CHECK-NEXT:   AddressAlignment: 1
// CHECK-NEXT:   EntrySize: 0
// CHECK-NEXT:   SectionData (
// CHECK-NEXT:     0000: 03050102
// CHECK-NEXT:   )
// CHECK-NEXT: }
// CHECK-NEXT: Section {
// CHECK-NEXT:   Index: 4
// CHECK-NEXT:   Name: .symtab

// CHECK:        Name: local
// CHECK-NEXT:   Value:
// CHECK-NEXT:   Size:
// CHECK-NEXT:   Binding:
// CHECK-NEXT:   Type:
// CHECK-NEXT:   Other:
// CHECK-NEXT:   Section: [[SEC:.*]]
// CHECK-NEXT: }
// CHECK-NEXT: Symbol {
// CHECK-NEXT:   Name:
// CHECK-NEXT:   Value:
// CHECK-NEXT:   Size:
// CHECK-NEXT:   Binding:
// CHECK-NEXT:   Type:
// CHECK-NEXT:   Other:
// CHECK-NEXT:   Section: [[SEC]]
// CHECK-NEXT: }
// CHECK-NEXT: Symbol {
// CHECK-NEXT:   Name: g1
// CHECK-NEXT:   Value:
// CHECK-NEXT:   Size:
// CHECK-NEXT:   Binding:
// CHECK-NEXT:   Type:
// CHECK-NEXT:   Other:
// CHECK-NEXT:   Section:
// CHECK-NEXT: }
// CHECK-NEXT: Symbol {
// CHECK-NEXT:   Name: g2
// CHECK-NEXT:   Value:
// CHECK-NEXT:   Size:
// CHECK-NEXT:   Binding:
// CHECK-NEXT:   Type:
// CHECK-NEXT:   Other:
// CHECK-NEXT:   Section:
// CHECK-NEXT: }
// CHECK-NEXT: Symbol {
// CHECK-NEXT:   Name: g3

// CHECK:      Addrsig [
// CHECK-NEXT:   Sym: g1 (3)
// CHECK-NEXT:   Sym: g3 (5)
// CHECK-NEXT:   Sym: local (1)
// CHECK-NEXT:   Sym:  (2)
// CHECK-NEXT: ]

// ASM: .addrsig
.addrsig
// ASM: .addrsig_sym g1
.addrsig_sym g1
.globl g2
// ASM: .addrsig_sym g3
.addrsig_sym g3
// ASM: .addrsig_sym local
.addrsig_sym local
// ASM: .addrsig_sym .Llocal
.addrsig_sym .Llocal

local:
.Llocal:

// DWO-NOT: .llvm_addrsig
