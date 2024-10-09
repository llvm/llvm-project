; RUN: clang -o %t %s
; RUN: %t | FileCheck %s
; CHECK: setjmp has been called
; CHECK-NEXT: Calling function foo
; CHECK-NEXT: Calling longjmp from inside function foo
; CHECK-NEXT: longjmp has been called
; CHECK-NEXT: Performing function recover
; ModuleID = 'builtin-setjmp-longjmp-02.c'
source_filename = "builtin-setjmp-longjmp-02.c"
target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-unknown-linux-gnu"

@buf = dso_local global [20 x ptr] zeroinitializer, align 8
@str = private unnamed_addr constant [41 x i8] c"Calling longjmp from inside function foo\00", align 1
@str.6 = private unnamed_addr constant [28 x i8] c"Performing function recover\00", align 1
@str.7 = private unnamed_addr constant [23 x i8] c"setjmp has been called\00", align 1
@str.8 = private unnamed_addr constant [21 x i8] c"Calling function foo\00", align 1
@str.10 = private unnamed_addr constant [24 x i8] c"longjmp has been called\00", align 1

; Function Attrs: noreturn nounwind
define dso_local void @foo() local_unnamed_addr #0 {
entry:
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf)
  unreachable
}

; Function Attrs: noreturn nounwind
declare void @llvm.eh.sjlj.longjmp(ptr) #1

; Function Attrs: nofree nounwind
define dso_local void @recover() local_unnamed_addr #2 {
entry:
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.6)
  ret void
}

; Function Attrs: noreturn nounwind
define dso_local noundef signext i32 @main() local_unnamed_addr #0 {
entry:
  %0 = tail call ptr @llvm.frameaddress.p0(i32 0)
  store ptr %0, ptr @buf, align 8
  %1 = tail call ptr @llvm.stacksave.p0()
  store ptr %1, ptr getelementptr inbounds (i8, ptr @buf, i64 16), align 8
  %2 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf)
  %cmp.not = icmp eq i32 %2, 0
  br i1 %cmp.not, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %puts6 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.10)
  tail call void @recover()
  tail call void @exit(i32 noundef signext 1) #8
  unreachable

if.end:                                           ; preds = %entry
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.7)
  %puts4 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.8)
  %puts.i = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf)
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.frameaddress.p0(i32 immarg) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #4

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr) #5

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef signext) local_unnamed_addr #6

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #7

attributes #0 = { noreturn nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { noreturn nounwind }
attributes #2 = { nofree nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #5 = { nounwind }
attributes #6 = { nofree noreturn nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #7 = { nofree nounwind }
attributes #8 = { cold noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git 19f04e908667aade0efe2de9ae705baaf68c0ce2)"}
