; RUN: not llc --mtriple=loongarch32 < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 < %s 2>&1 | FileCheck %s

declare void @llvm.loongarch.dbar(i32)
declare void @llvm.loongarch.ibar(i32)
declare void @llvm.loongarch.break(i32)
declare void @llvm.loongarch.syscall(i32)

define void @dbar_imm_out_of_hi_range() nounwind {
; CHECK: argument to '__builtin_loongarch_dbar' out of range
entry:
  call void @llvm.loongarch.dbar(i32 32769)
  ret void
}

define void @dbar_imm_out_of_lo_range() nounwind {
; CHECK: argument to '__builtin_loongarch_dbar' out of range
entry:
  call void @llvm.loongarch.dbar(i32 -1)
  ret void
}

define void @ibar_imm_out_of_hi_range() nounwind {
; CHECK: argument to '__builtin_loongarch_ibar' out of range
entry:
  call void @llvm.loongarch.ibar(i32 32769)
  ret void
}

define void @ibar_imm_out_of_lo_range() nounwind {
; CHECK: argument to '__builtin_loongarch_ibar' out of range
entry:
  call void @llvm.loongarch.ibar(i32 -1)
  ret void
}

define void @break_imm_out_of_hi_range() nounwind {
; CHECK: argument to '__builtin_loongarch_break' out of range
entry:
  call void @llvm.loongarch.break(i32 32769)
  ret void
}

define void @break_imm_out_of_lo_range() nounwind {
; CHECK: argument to '__builtin_loongarch_break' out of range
entry:
  call void @llvm.loongarch.break(i32 -1)
  ret void
}

define void @syscall_imm_out_of_hi_range() nounwind {
; CHECK: argument to '__builtin_loongarch_syscall' out of range
entry:
  call void @llvm.loongarch.syscall(i32 32769)
  ret void
}

define void @syscall_imm_out_of_lo_range() nounwind {
; CHECK: argument to '__builtin_loongarch_syscall' out of range
entry:
  call void @llvm.loongarch.syscall(i32 -1)
  ret void
}
