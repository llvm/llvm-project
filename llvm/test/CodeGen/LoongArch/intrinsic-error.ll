; RUN: not llc --mtriple=loongarch32 --disable-verify < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --disable-verify < %s 2>&1 | FileCheck %s

define void @dbar_not_constant(i32 %x) nounwind {
; CHECK: argument to '__builtin_loongarch_dbar' must be a constant integer
entry:
  call void @llvm.loongarch.dbar(i32 %x)
  ret void
}

define void @dbar_imm_out_of_range() nounwind {
; CHECK: argument to '__builtin_loongarch_dbar' out of range
entry:
  call void @llvm.loongarch.dbar(i32 32769)
  ret void
}

declare void @llvm.loongarch.dbar(i32)
