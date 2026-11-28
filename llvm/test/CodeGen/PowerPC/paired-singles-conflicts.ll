; RUN: not llc -mtriple=powerpc64-unknown-linux-gnu -mattr=+paired-singles \
; RUN:   < %s 2>&1 | FileCheck --check-prefix=CHECK-PPC64 %s
; RUN: not llc -mtriple=powerpcle-unknown-linux-gnu -mattr=+paired-singles \
; RUN:   < %s 2>&1 | FileCheck --check-prefix=CHECK-LE %s
; RUN: not llc -mtriple=powerpc-unknown-linux-gnu -mattr=+paired-singles,+altivec \
; RUN:   < %s 2>&1 | FileCheck --check-prefix=CHECK-ALTIVEC %s

; CHECK-PPC64: Paired Singles are only supported for 32-bit targets.
; CHECK-LE: Paired Singles are only supported for Big-Endian targets.
; CHECK-ALTIVEC: Paired Singles and Altivec cannot both be enabled.

define void @f() {
  ret void
}
