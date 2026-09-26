; The "target-abi" module flag selects the ABI used for codegen. N32
; adjusts the stack with addiu while N64 uses daddiu, and the two ABIs
; are recorded in different .mdebug sections.

; RUN: split-file %s %t

; mips64 defaults to N64; an n32 module flag overrides that.
; RUN: llc -mtriple=mips64 -mcpu=mips64r2 < %t/n32.ll | FileCheck --check-prefix=N32 %s
; RUN: llc -mtriple=mips64 -mcpu=mips64r2 < %t/default.ll | FileCheck --check-prefix=N64 %s

; A matching -target-abi option is accepted.
; RUN: llc -mtriple=mips64 -mcpu=mips64r2 -target-abi=n32 < %t/n32.ll | FileCheck --check-prefix=N32 %s

; A conflicting -target-abi option is rejected.
; RUN: not llc -mtriple=mips64 -mcpu=mips64r2 -target-abi=n64 < %t/n32.ll 2>&1 | FileCheck --check-prefix=CONFLICT %s

; N32: .section .mdebug.abiN32
; N32: {{[[:space:]]}}addiu $sp, $sp, -16

; N64: .section .mdebug.abi64
; N64: {{[[:space:]]}}daddiu $sp, $sp, -16

; CONFLICT: -target-abi option != target-abi module flag

;--- n32.ll
declare void @g(i32)

define void @f(i32 %x) {
  call void @g(i32 %x)
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", !"n32"}

;--- default.ll
declare void @g(i32)

define void @f(i32 %x) {
  call void @g(i32 %x)
  ret void
}
