# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S --symbols --sd --cg-profile - | FileCheck %s

  .section .test,"aw",@progbits
a: .word b

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
# CHECK-NEXT: Type: SHT_LLVM_CALL_GRAPH_PROFILE (0x6FFF4C02)
# CHECK-NEXT: Flags [ (0x80000000)
# CHECK-NEXT: SHF_EXCLUDE (0x80000000)
# CHECK-NEXT: ]
# CHECK-NEXT: Address:
# CHECK-NEXT: Offset:
# CHECK-NEXT: Size: 64
# CHECK-NEXT: Link: 6
# CHECK-NEXT: Info: 0
# CHECK-NEXT: AddressAlignment: 1
# CHECK-NEXT: EntrySize: 16
# CHECK-NEXT: SectionData (
# CHECK-NEXT:   0000: 02000000 05000000 20000000 00000000
# CHECK-NEXT:   0010: 07000000 02000000 0B000000 00000000
# CHECK-NEXT:   0020: 06000000 03000000 14000000 00000000
# CHECK-NEXT:   0030: 01000000 05000000 2A000000 00000000
# CHECK-NEXT: )

# CHECK: Symbols [
# CHECK:      Name: a
# CHECK-NEXT: Value:
# CHECK-NEXT: Size:
# CHECK-NEXT: Binding: Local
# CHECK-NEXT: Type:
# CHECK-NEXT: Other:
# CHECK-NEXT: Section: .test
# CHECK:      Name: late2
# CHECK-NEXT: Value:
# CHECK-NEXT: Size:
# CHECK-NEXT: Binding: Local
# CHECK-NEXT: Type:
# CHECK-NEXT: Other:
# CHECK-NEXT: Section: .test
# CHECK:      Name: late3
# CHECK-NEXT: Value:
# CHECK-NEXT: Size:
# CHECK-NEXT: Binding: Local
# CHECK-NEXT: Type:
# CHECK-NEXT: Other:
# CHECK-NEXT: Section: .test
# CHECK:      Name: b
# CHECK-NEXT: Value:
# CHECK-NEXT: Size:
# CHECK-NEXT: Binding: Global
# CHECK-NEXT: Type:
# CHECK-NEXT: Other:
# CHECK-NEXT: Section: Undefined
# CHECK:      Name: late
# CHECK-NEXT: Value:
# CHECK-NEXT: Size:
# CHECK-NEXT: Binding: Global
# CHECK-NEXT: Type:
# CHECK-NEXT: Other:
# CHECK-NEXT: Section: .test
# CHECK:      Name: freq
# CHECK-NEXT: Value:
# CHECK-NEXT: Size:
# CHECK-NEXT: Binding: Weak
# CHECK-NEXT: Type:
# CHECK-NEXT: Other:
# CHECK-NEXT: Section: Undefined
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
# CHECK-NEXT:   CGProfileEntry {
# CHECK-NEXT:     From:
# CHECK-NEXT:     To: b
# CHECK-NEXT:     Weight: 42
# CHECK-NEXT:   }
# CHECK-NEXT: ]
