; RUN: llc < %s | FileCheck %s
; ModuleID = 'setjmp.c'
source_filename = "setjmp.c"
target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-unknown-linux-gnu"

@buf = dso_local global [20 x ptr] zeroinitializer, align 8

; Function Attrs: nounwind
define dso_local signext range(i32 0, 2) i32 @main(i32 noundef signext %argc, ptr nocapture noundef readnone %argv) local_unnamed_addr #0 {
entry:
  %0 = tail call ptr @llvm.frameaddress.p0(i32 0)
  store ptr %0, ptr @buf, align 8
  %1 = tail call ptr @llvm.stacksave.p0()
  store ptr %1, ptr getelementptr inbounds (i8, ptr @buf, i64 24), align 8
; CHECK: stmg    %r6, %r15, 48(%r15)
; CHECK: lgr     %r1, %r15
; CHECK: aghi     %r15, -224
; CHECK: stg     %r1, 0(%r15)
; CHECK: std     %f8, 216(%r15)
; CHECK: std     %f9, 208(%r15)
; CHECK: std     %f10, 200(%r15)
; CHECK: std     %f11, 192(%r15)
; CHECK: std     %f12, 184(%r15)
; CHECK: std     %f13, 176(%r15)
; CHECK: std     %f14, 168(%r15)
; CHECK: std     %f15, 160(%r15)
; CHECK: la      %r0, 224(%r15)
; CHECK: stgrl   %r0, buf
; CHECK: stgrl   %r15, buf+24
; CHECK: larl    %r1, buf
; CHECK: larl    %r0, .LBB0_2
; CHECK: stg     %r0, 8(%r1)
; CHECK: lg      %r0, 0(%r15)
; CHECK: stg     %r0, 16(%r1)
; CHECK: stg     %r13, 32(%r1)
  %2 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf)
  %tobool.not = icmp eq i32 %2, 0
  br i1 %tobool.not, label %if.end, label %return

if.end:                                           ; preds = %entry
  tail call void @foo() #3
  br label %return

return:                                           ; preds = %entry, %if.end
  %retval.0 = phi i32 [ 1, %if.end ], [ 0, %entry ]
  ret i32 %retval.0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.frameaddress.p0(i32 immarg) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #2

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr) #3

declare void @foo() local_unnamed_addr #4

attributes #0 = { nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #3 = { nounwind }
attributes #4 = { "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git 19f04e908667aade0efe2de9ae705baaf68c0ce2)"}
