; Check that the "target-abi" module flag selects the PPC ELF ABI, so the ABI
; is taken from the IR when no -target-abi option is given. The flag overrides
; the triple default in both directions (.abiversion 2 is emitted for ELFv2).

; RUN: split-file %s %t

; powerpc64 big-endian defaults to ELFv1; an elfv2 module flag overrides that.
; RUN: llc -mtriple=powerpc64-unknown-linux < %t/elfv2.ll | FileCheck %s --check-prefix=ELFv2

; powerpc64le defaults to ELFv2; an elfv1 module flag overrides that.
; RUN: llc -mtriple=powerpc64le-unknown-linux < %t/elfv1.ll | FileCheck %s --check-prefix=ELFv1

; A matching -target-abi option is accepted.
; RUN: llc -mtriple=powerpc64-unknown-linux -target-abi elfv2 < %t/elfv2.ll | FileCheck %s --check-prefix=ELFv2

; A conflicting -target-abi option is rejected.
; RUN: not llc -mtriple=powerpc64-unknown-linux -target-abi elfv1 < %t/elfv2.ll 2>&1 | FileCheck %s --check-prefix=CONFLICT

; ELFv2: .abiversion 2
; ELFv1-NOT: .abiversion 2
; CONFLICT: -target-abi option != target-abi module flag

;--- elfv1.ll
define void @f() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", !"elfv1"}

;--- elfv2.ll
define void @g() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", !"elfv2"}
