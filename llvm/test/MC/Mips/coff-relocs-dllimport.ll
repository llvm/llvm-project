; RUN: llc -mtriple mipsel-windows -filetype obj < %s | llvm-objdump --reloc - | FileCheck %s

declare dllimport void @fun()

define void @use() nounwind {
; CHECK: 00000008 IMAGE_REL_MIPS_REFHI     __imp_fun
; CHECK: 0000000c IMAGE_REL_MIPS_REFLO     __imp_fun
  call void() @fun()

  ret void
}
