# RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s -o %t
# RUN: llvm-readobj -S --symbols --sd --cg-profile %t | FileCheck %s

  .section .test,"w"
a:

  .cg_profile a, b, 32
  .cg_profile freq, a, 11
  .cg_profile late, late2, 20
  .cg_profile .L.local, b, 42

	.globl late
late:
late2: .word 0
late3:
.L.local:

# CHECK:      Name: .llvm.call-graph-profile
# CHECK-NEXT: VirtualSize:
# CHECK-NEXT: VirtualAddress:
# CHECK-NEXT: RawDataSize: 48
# CHECK-NEXT: PointerToRawData:
# CHECK-NEXT: PointerToRelocations:
# CHECK-NEXT: PointerToLineNumbers:
# CHECK-NEXT: RelocationCount:
# CHECK-NEXT: LineNumberCount:
# CHECK-NEXT: Characteristics [ (0x100800)
# CHECK-NEXT:   IMAGE_SCN_ALIGN_1BYTES (0x100000)
# CHECK-NEXT:   IMAGE_SCN_LNK_REMOVE (0x800)
# CHECK-NEXT: ]
# CHECK-NEXT: SectionData (
# CHECK-NEXT:   0000: 0A000000 0E000000 20000000 00000000
# CHECK-NEXT:   0010: 11000000 0A000000 0B000000 00000000
# CHECK-NEXT:   0020: 0B000000 0C000000 14000000 00000000
# CHECK-NEXT: )

# CHECK: Symbols [
# CHECK:      Name: a
# CHECK-NEXT: Value:
# CHECK-NEXT: Section: .test
# CHECK-NEXT: BaseType:
# CHECK-NEXT: ComplexType:
# CHECK-NEXT: StorageClass: Static
# CHECK-NEXT: AuxSymbolCount:
# CHECK:      Name: late
# CHECK-NEXT: Value:
# CHECK-NEXT: Section: .test
# CHECK-NEXT: BaseType:
# CHECK-NEXT: ComplexType:
# CHECK-NEXT: StorageClass: External
# CHECK-NEXT: AuxSymbolCount:
# CHECK:      Name: late2
# CHECK-NEXT: Value:
# CHECK-NEXT: Section: .test
# CHECK-NEXT: BaseType:
# CHECK-NEXT: ComplexType:
# CHECK-NEXT: StorageClass: Static
# CHECK-NEXT: AuxSymbolCount:
# CHECK:      Name: late3
# CHECK-NEXT: Value:
# CHECK-NEXT: Section: .test
# CHECK-NEXT: BaseType:
# CHECK-NEXT: ComplexType:
# CHECK-NEXT: StorageClass: Static
# CHECK-NEXT: AuxSymbolCount:
# CHECK:      Name: b
# CHECK-NEXT: Value:
# CHECK-NEXT: Section: IMAGE_SYM_UNDEFINED
# CHECK-NEXT: BaseType:
# CHECK-NEXT: ComplexType:
# CHECK-NEXT: StorageClass: WeakExternal
# CHECK-NEXT: AuxSymbolCount: 1
# CHECK-NEXT: AuxWeakExternal {
# CHECK-NEXT:   Linked: .weak.b.default.late
# CHECK-NEXT:   Search: Alias
# CHECK-NEXT: }
# CHECK:      Name: .weak.b.default.late
# CHECK-NEXT: Value:
# CHECK-NEXT: Section: IMAGE_SYM_ABSOLUTE
# CHECK-NEXT: BaseType:
# CHECK-NEXT: ComplexType:
# CHECK-NEXT: StorageClass: External
# CHECK-NEXT: AuxSymbolCount: 0
# CHECK:      Name: freq
# CHECK-NEXT: Value:
# CHECK-NEXT: Section: IMAGE_SYM_UNDEFINED
# CHECK-NEXT: BaseType:
# CHECK-NEXT: ComplexType:
# CHECK-NEXT: StorageClass: WeakExternal
# CHECK-NEXT: AuxSymbolCount: 1
# CHECK-NEXT: AuxWeakExternal {
# CHECK-NEXT:   Linked: .weak.freq.default.late
# CHECK-NEXT:   Search: Alias
# CHECK-NEXT: }
# CHECK:      Name: .weak.freq.default.late
# CHECK-NEXT: Value:
# CHECK-NEXT: Section: IMAGE_SYM_ABSOLUTE
# CHECK-NEXT: BaseType:
# CHECK-NEXT: ComplexType:
# CHECK-NEXT: StorageClass: External
# CHECK-NEXT: AuxSymbolCount: 0

# CHECK:      CGProfile [
# CHECK-NEXT:   CGProfileEntry {
# CHECK-NEXT:     From: a
# CHECK-NEXT:     To: b
# CHECK-NEXT:     Weight: 32
# CHECK-NEXT:   }
# CHECK-NEXT:   CGProfileEntry {
# CHECK-NEXT:     From: freq
# CHECK-NEXT:     To: a
# CHECK-NEXT:     Weight: 11
# CHECK-NEXT:   }
# CHECK-NEXT:   CGProfileEntry {
# CHECK-NEXT:     From: late
# CHECK-NEXT:     To: late2
# CHECK-NEXT:     Weight: 20
# CHECK-NEXT:   }
# CHECK-NEXT: ]
