; RUN: llc -mtriple=riscv32-unknown-linux-gnu -verify-machineinstrs -o - %s \
; RUN: | FileCheck --check-prefix=INITARRAY %s
; RUN: llc -mtriple=riscv32-unknown-elf -verify-machineinstrs -o - %s \
; RUN: | FileCheck --check-prefix=INITARRAY %s
; RUN: llc -mtriple=riscv64-unknown-linux-gnu -verify-machineinstrs -o - %s \
; RUN: | FileCheck --check-prefix=INITARRAY %s
; RUN: llc -mtriple=riscv64-unknown-elf -verify-machineinstrs -o - %s \
; RUN: | FileCheck --check-prefix=INITARRAY %s

; RUN: llc -mtriple=riscv32-unknown-linux-gnu -verify-machineinstrs -use-ctors -o - %s \
; RUN: | FileCheck --check-prefix=CTOR %s
; RUN: llc -mtriple=riscv32-unknown-elf -verify-machineinstrs -use-ctors -o - %s \
; RUN: | FileCheck --check-prefix=CTOR %s
; RUN: llc -mtriple=riscv64-unknown-linux-gnu -verify-machineinstrs -use-ctors -o - %s \
; RUN: | FileCheck --check-prefix=CTOR %s
; RUN: llc -mtriple=riscv64-unknown-elf -verify-machineinstrs -use-ctors -o - %s \
; RUN: | FileCheck --check-prefix=CTOR %s

define internal void @_GLOBAL__I_a() section ".text.startup" {
  ret void
}

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__I_a, ptr null }]

;INITARRAY: section .init_array
;INITARRAY-NOT: .section    .ctors

;CTOR: .section .ctors
;CTOR-NOT:  section .init_array
