; RUN: llc %s -mtriple=armv7-linux-gnueabi -relocation-model=pic \
; RUN: -filetype=obj -o - | elf-dump --dump-section-data | FileCheck %s


@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 65535, void ()* @f }]

define void @f() {
  ret void
}

; Check for a relocation of type R_ARM_TARGET1.
; CHECK: ('r_type', 0x26)
