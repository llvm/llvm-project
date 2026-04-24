; Check that the backend doesn't crash.
; RUN: llc -mtriple=x86_64-pc-freebsd %s -o - | FileCheck %s

@__stack_chk_guard = internal global [8 x i64] zeroinitializer, align 16

declare void @llvm.ssp.protected(ptr nocapture) nofree nosync nounwind memory(none)

define void @f() sspstrong {
  %tbl = alloca [4 x i64], align 16
  call void @llvm.ssp.protected(ptr %tbl)
  ret void
}

; CHECK:  movq  __stack_chk_guard(%rip), %rax
; CHECK:  movq  __stack_chk_guard(%rip), %rax
; CHECK:  .type __stack_chk_guard,@object
; CHECK:  .local  __stack_chk_guard
; CHECK:  .comm __stack_chk_guard,64,16
