; RUN: llc -mtriple=aarch64-none-linux-gnu -verify-machineinstrs -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-none-elf -verify-machineinstrs -o - %s | FileCheck %s

define internal void @_GLOBAL__I_a() section ".text.startup" {
  ret void
}

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__I_a, ptr null }]

; CHECK: .section .init_array
