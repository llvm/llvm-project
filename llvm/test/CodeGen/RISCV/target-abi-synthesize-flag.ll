; Check behavior of the -target-abi command-line option; it should
; synthesize the "target-abi" module flag, unless one is already
; present. A conflict is diagnosed by verifyOptionsConsistency.

; RUN: split-file %s %t

; -target-abi synthesizes the flag and selects the ABI, overriding the
; extension default (lp64d).
; RUN: llc -mtriple=riscv64 -mattr=+d -target-abi lp64f -filetype=obj < %t/none.ll | llvm-readelf -h - | FileCheck %s --check-prefix=LP64F

; Without -target-abi, the extension-default ABI is used.
; RUN: llc -mtriple=riscv64 -mattr=+d -filetype=obj < %t/none.ll | llvm-readelf -h - | FileCheck %s --check-prefix=LP64D

; An unrecognized ABI name in the module flag is an error
; RUN: not llc -mtriple=riscv64 -mattr=+d -filetype=null < %t/bogus.ll 2>&1 | FileCheck %s --check-prefix=BADABI

; -target-abi matching an existing in-IR flag is accepted.
; RUN: llc -mtriple=riscv64 -mattr=+d -target-abi lp64f -filetype=obj < %t/lp64f.ll | llvm-readelf -h - | FileCheck %s --check-prefix=LP64F

; -target-abi conflicting with an existing in-IR flag is an error.
; RUN: not llc -mtriple=riscv64 -mattr=+d -target-abi lp64d -filetype=null < %t/lp64f.ll 2>&1 | FileCheck %s --check-prefix=CONFLICT

; LP64F: Flags: 0x2, single-float ABI
; LP64D: Flags: 0x4, double-float ABI
; BADABI: error: 'bogus' is not a recognized ABI for this target
; CONFLICT: error: -target-abi option != target-abi module flag

;--- none.ll
define void @f() {
  ret void
}

;--- lp64f.ll
define void @f() {
  ret void
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", !"lp64f"}

;--- bogus.ll
define void @f() {
  ret void
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", !"bogus"}
