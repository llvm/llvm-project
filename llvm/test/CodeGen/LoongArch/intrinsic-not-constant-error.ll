; RUN: not llc --mtriple=loongarch32 < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 < %s 2>&1 | FileCheck %s

declare void @llvm.loongarch.dbar(i32)
declare void @llvm.loongarch.ibar(i32)
declare void @llvm.loongarch.break(i32)
declare void @llvm.loongarch.movgr2fcsr(i32, i32)
declare i32 @llvm.loongarch.movfcsr2gr(i32)
declare void @llvm.loongarch.syscall(i32)

define void @dbar_not_constant(i32 %x) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.dbar(i32 %x)
  ret void
}

define void @ibar(i32 %x) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.ibar(i32 %x)
  ret void
}

define void @break(i32 %x) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.break(i32 %x)
  ret void
}

define void @movgr2fcsr(i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.movgr2fcsr(i32 %a, i32 %a)
  ret void
}

define i32 @movfcsr2gr(i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call i32 @llvm.loongarch.movfcsr2gr(i32 %a)
  ret i32 %res
}

define void @syscall(i32 %x) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.syscall(i32 %x)
  ret void
}
