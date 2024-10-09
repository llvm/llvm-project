; RUN: llc < %s | FileCheck %s
; ModuleID = 'longjmp.c'
source_filename = "longjmp.c"
target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-unknown-linux-gnu"

@buf = dso_local global [20 x ptr] zeroinitializer, align 8

; Function Attrs: noreturn nounwind
define dso_local void @foo() local_unnamed_addr #0 {
entry:
; CHECK: stmg    %r13, %r15, 104(%r15)
; CHECK: larl    %r1, buf
; CHECK: lg      %r2, 8(%r1)
; CHECK: lg      %r13, 32(%r1)
; CHECK: lg      %r15, 24(%r1)
; CHECK: lg      %r1, 16(%r1)
; CHECK: stg     %r1, 0(%r15)
; CHECK: br      %r2
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf)
  unreachable
}

; Function Attrs: noreturn nounwind
declare void @llvm.eh.sjlj.longjmp(ptr) #1

attributes #0 = { noreturn nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git 19f04e908667aade0efe2de9ae705baaf68c0ce2)"}
