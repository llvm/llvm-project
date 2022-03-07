; RUN: llc -mtriple=arm64-apple-macosx %s -filetype=obj -o - | llvm-objdump --unwind-info - | FileCheck %s

; CHECK: Contents of __compact_unwind section
; CHECK: compact encoding: 0x02021010

; 0x02|021|010 => frameless, stack size 0x21 * 16 = 528, x27 & x28 saved

define void @func() {
  alloca i8, i32 512
  ret void
}
