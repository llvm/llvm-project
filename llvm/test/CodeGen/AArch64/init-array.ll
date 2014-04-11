; RUN: llc -mtriple=aarch64-none-linux-gnu -verify-machineinstrs -use-init-array < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-none-none-eabi -verify-machineinstrs -use-init-array < %s | FileCheck %s

define internal void @_GLOBAL__I_a() section ".text.startup" {
  ret void
}

@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 65535, void ()* @_GLOBAL__I_a }]

; CHECK: .section .init_array
