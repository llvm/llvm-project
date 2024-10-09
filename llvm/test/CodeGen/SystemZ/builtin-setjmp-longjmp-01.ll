; RUN: clang -o %t %s
; RUN: %t | FileCheck %s
; CHECK: call foo
; CHECK: return 
; ModuleID = 'builtin-setjmp-longjmp-01.c'
source_filename = "builtin-setjmp-longjmp-01.c"
target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-unknown-linux-gnu"

@buf = dso_local global [20 x ptr] zeroinitializer, align 8
@str = private unnamed_addr constant [9 x i8] c"call foo\00", align 1
@str.2 = private unnamed_addr constant [7 x i8] c"return\00", align 1

; Function Attrs: noreturn nounwind
define dso_local void @foo() local_unnamed_addr #0 {
entry:
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf)
  unreachable
}

; Function Attrs: noreturn nounwind
declare void @llvm.eh.sjlj.longjmp(ptr) #1

; Function Attrs: nounwind
define dso_local noundef signext range(i32 0, 2) i32 @main(i32 noundef signext %argc, ptr nocapture noundef readnone %argv) local_unnamed_addr #2 {
entry:
  %0 = tail call ptr @llvm.frameaddress.p0(i32 0)
  store ptr %0, ptr @buf, align 8
  %1 = tail call ptr @llvm.stacksave.p0()
  store ptr %1, ptr getelementptr inbounds (i8, ptr @buf, i64 16), align 8
  %2 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf)
  %tobool.not = icmp eq i32 %2, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %puts2 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.2)
  ret i32 0

if.end:                                           ; preds = %entry
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf)
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.frameaddress.p0(i32 immarg) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #4

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr) #5

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #6

attributes #0 = { noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { noreturn nounwind }
attributes #2 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #5 = { nounwind }
attributes #6 = { nofree nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git 19f04e908667aade0efe2de9ae705baaf68c0ce2)"}
