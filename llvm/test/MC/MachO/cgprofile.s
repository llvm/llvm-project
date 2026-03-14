# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t
# RUN: llvm-readobj -S --symbols --sd --cg-profile %t | FileCheck %s

  .section __TEXT,__text
a:

  .cg_profile a, b, 32
  .cg_profile freq, a, 11
  .cg_profile late, late2, 20
  .cg_profile L.local, b, 42

	.globl late
late:
late2: .word 0
late3:
L.local:


# CHECK:      Name: __cg_profile
# CHECK-NEXT: Segment: __LLVM
# CHECK-NEXT: Address:
# CHECK-NEXT: Size: 0x30
# CHECK:      SectionData (
# CHECK-NEXT:   0000: 00000000 04000000 20000000 00000000
# CHECK-NEXT:   0010: 05000000 00000000 0B000000 00000000
# CHECK-NEXT:   0020: 03000000 01000000 14000000 00000000
# CHECK-NEXT: )

# CHECK:        CGProfile [
# CHECK-NEXT:   CGProfileEntry {
# CHECK-NEXT:     From: a (0)
# CHECK-NEXT:     To: b (4)
# CHECK-NEXT:     Weight: 32
# CHECK-NEXT:   }
# CHECK-NEXT:   CGProfileEntry {
# CHECK-NEXT:     From: freq (5)
# CHECK-NEXT:     To: a (0)
# CHECK-NEXT:     Weight: 11
# CHECK-NEXT:   }
# CHECK-NEXT:   CGProfileEntry {
# CHECK-NEXT:     From: late (3)
# CHECK-NEXT:     To: late2 (1)
# CHECK-NEXT:     Weight: 20
# CHECK-NEXT:   }
# CHECK-NEXT: ]
