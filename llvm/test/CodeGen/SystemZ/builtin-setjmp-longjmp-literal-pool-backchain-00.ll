; -mbackchain option
; Test output for inline Literal Pool.
; RUN: clang -mbackchain -o %t %s
; RUN: %t | FileCheck %s
; CHECK: value_ptr is 954219
; CHECK: value_ptr is 954219
; CHECK: value_ptr is 420420

; ModuleID = 'builtin-setjmp-longjmp-literal-pool-00.c'
source_filename = "builtin-setjmp-longjmp-literal-pool-00.c"
target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-unknown-linux-gnu"

module asm ".LC101:"
module asm ".long 123454"
module asm ".long 451233"
module asm ".long 954219"
module asm ".long 466232"
module asm ".long 955551"
module asm ".long 687823"
module asm ".long 555123"
module asm ".long 777723"
module asm ".long 985473"
module asm ".long 190346"
module asm ".long 420420"
module asm ".long 732972"
module asm ".long 971166"
module asm ".long 123454"
module asm ".long 451233"
module asm ".long 954219"
module asm ".long 466232"
module asm ".long 955551"
module asm ".long 687823"
module asm ".long 555123"

@buf = dso_local global [20 x ptr] zeroinitializer, align 8
@.str = private unnamed_addr constant [17 x i8] c"value_ptr is %d\0A\00", align 2

; Function Attrs: noreturn nounwind
define dso_local void @foo() local_unnamed_addr #0 {
entry:
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf)
  unreachable
}

; Function Attrs: noreturn nounwind
declare void @llvm.eh.sjlj.longjmp(ptr) #1

; Function Attrs: nounwind
define dso_local noundef signext i32 @main(i32 noundef signext %argc, ptr nocapture noundef readnone %argv) local_unnamed_addr #2 {
entry:
  %0 = tail call ptr asm sideeffect "larl $0, .LC101", "={r13}"() #3, !srcloc !4
  %add.ptr = getelementptr inbounds i8, ptr %0, i64 8
  %1 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf)
  %cmp = icmp eq i32 %1, 0
  %2 = load i32, ptr %add.ptr, align 4, !tbaa !5
  %call = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef signext %2)
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf)
  unreachable

if.else:                                          ; preds = %entry
  %add.ptr2 = getelementptr inbounds i8, ptr %0, i64 40
  %3 = load i32, ptr %add.ptr2, align 4, !tbaa !5
  %call3 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef signext %3)
  ret i32 0
}

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr) #3

; Function Attrs: nofree nounwind
declare noundef signext i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #4

attributes #0 = { noreturn nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { noreturn nounwind }
attributes #2 = { nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #3 = { nounwind }
attributes #4 = { nofree nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git 79880371396d6e486bf6bacd6c4087ebdac591f8)"}
!4 = !{i64 751}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
