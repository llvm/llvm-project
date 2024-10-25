; RUN: llc < %s | FileCheck %s
; ModuleID = 'builtin-setjmp-spills-int-02.c'
source_filename = "builtin-setjmp-spills-int-02.c"
target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-unknown-linux-gnu"

@buf = dso_local global [10 x ptr] zeroinitializer, align 8
@t = dso_local global i32 0, align 4
@s = dso_local global i32 0, align 4
@r = dso_local global i32 0, align 4
@q = dso_local global i32 0, align 4
@p = dso_local global i32 0, align 4
@o = dso_local global i32 0, align 4
@n = dso_local global i32 0, align 4
@m = dso_local global i32 0, align 4
@l = dso_local global i32 0, align 4
@k = dso_local global i32 0, align 4
@j = dso_local global i32 0, align 4
@i = dso_local global i32 0, align 4
@h = dso_local global i32 0, align 4
@g = dso_local global i32 0, align 4
@f = dso_local global i32 0, align 4
@e = dso_local global i32 0, align 4
@d = dso_local global i32 0, align 4
@c = dso_local global i32 0, align 4
@b = dso_local global i32 0, align 4
@a = dso_local global i32 0, align 4
@str = private unnamed_addr constant [40 x i8] c"First time through, all variables are 1\00", align 1

; Function Attrs: nounwind
define dso_local signext i32 @func() local_unnamed_addr #0 {
entry:
  %0 = tail call ptr @llvm.frameaddress.p0(i32 0)
  store ptr %0, ptr @buf, align 8
  %1 = tail call ptr @llvm.stacksave.p0()
  store ptr %1, ptr getelementptr inbounds (i8, ptr @buf, i64 16), align 8
; CHECK: stmg    %r6, %r15, 48(%r15)
; CHECK: ghi     %r15, -344
; CHECK: std     %f8, 336(%r15)
; CHECK: std     %f9, 328(%r15)
; CHECK: std     %f10, 320(%r15)
; CHECK: std     %f11, 312(%r15)
; CHECK: std     %f12, 304(%r15)
; CHECK: std     %f13, 296(%r15)
; CHECK: std     %f14, 288(%r15)
; CHECK: std     %f15, 280(%r15)
; CHECK: la      %r0, 344(%r15)
; CHECK: stgrl   %r0, buf
; CHECK: stgrl   %r15, buf+16
; CHECK: larl    %r1, buf
; CHECK: larl    %r0, .LBB0_2
; CHECK: stg     %r0, 8(%r1)
; CHECK: stg     %r15, 24(%r1)
  %2 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf)
  %cmp = icmp eq i32 %2, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 1, ptr @t, align 4, !tbaa !4
  store i32 1, ptr @s, align 4, !tbaa !4
  store i32 1, ptr @r, align 4, !tbaa !4
  store i32 1, ptr @q, align 4, !tbaa !4
  store i32 1, ptr @p, align 4, !tbaa !4
  store i32 1, ptr @o, align 4, !tbaa !4
  store i32 1, ptr @n, align 4, !tbaa !4
  store i32 1, ptr @m, align 4, !tbaa !4
  store i32 1, ptr @l, align 4, !tbaa !4
  store i32 1, ptr @k, align 4, !tbaa !4
  store i32 1, ptr @j, align 4, !tbaa !4
  store i32 1, ptr @i, align 4, !tbaa !4
  store i32 1, ptr @h, align 4, !tbaa !4
  store i32 1, ptr @g, align 4, !tbaa !4
  store i32 1, ptr @f, align 4, !tbaa !4
  store i32 1, ptr @e, align 4, !tbaa !4
  store i32 1, ptr @d, align 4, !tbaa !4
  store i32 1, ptr @c, align 4, !tbaa !4
  store i32 1, ptr @b, align 4, !tbaa !4
  store i32 1, ptr @a, align 4, !tbaa !4
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  tail call void @foo(ptr noundef nonnull @a, ptr noundef nonnull @b, ptr noundef nonnull @c, ptr noundef nonnull @d, ptr noundef nonnull @e, ptr noundef nonnull @f, ptr noundef nonnull @g, ptr noundef nonnull @h, ptr noundef nonnull @i, ptr noundef nonnull @j, ptr noundef nonnull @k, ptr noundef nonnull @l, ptr noundef nonnull @m, ptr noundef nonnull @n, ptr noundef nonnull @o, ptr noundef nonnull @p, ptr noundef nonnull @q, ptr noundef nonnull @r, ptr noundef nonnull @s, ptr noundef nonnull @t) #3
  %3 = load i32, ptr @a, align 4, !tbaa !4
  %4 = load i32, ptr @b, align 4, !tbaa !4
  %add = add nsw i32 %4, %3
  %5 = load i32, ptr @c, align 4, !tbaa !4
  %add1 = add nsw i32 %add, %5
  %6 = load i32, ptr @d, align 4, !tbaa !4
  %add2 = add nsw i32 %add1, %6
  %7 = load i32, ptr @e, align 4, !tbaa !4
  %add3 = add nsw i32 %add2, %7
  %8 = load i32, ptr @f, align 4, !tbaa !4
  %add4 = add nsw i32 %add3, %8
  %9 = load i32, ptr @g, align 4, !tbaa !4
  %add5 = add nsw i32 %add4, %9
  %10 = load i32, ptr @h, align 4, !tbaa !4
  %add6 = add nsw i32 %add5, %10
  %11 = load i32, ptr @i, align 4, !tbaa !4
  %add7 = add nsw i32 %add6, %11
  %12 = load i32, ptr @j, align 4, !tbaa !4
  %add8 = add nsw i32 %add7, %12
  %13 = load i32, ptr @k, align 4, !tbaa !4
  %add9 = add nsw i32 %add8, %13
  %14 = load i32, ptr @l, align 4, !tbaa !4
  %add10 = add nsw i32 %add9, %14
  %15 = load i32, ptr @m, align 4, !tbaa !4
  %add11 = add nsw i32 %add10, %15
  %16 = load i32, ptr @n, align 4, !tbaa !4
  %add12 = add nsw i32 %add11, %16
  %17 = load i32, ptr @o, align 4, !tbaa !4
  %add13 = add nsw i32 %add12, %17
  %18 = load i32, ptr @p, align 4, !tbaa !4
  %add14 = add nsw i32 %add13, %18
  %19 = load i32, ptr @q, align 4, !tbaa !4
  %add15 = add nsw i32 %add14, %19
  %20 = load i32, ptr @r, align 4, !tbaa !4
  %add16 = add nsw i32 %add15, %20
  %21 = load i32, ptr @s, align 4, !tbaa !4
  %add17 = add nsw i32 %add16, %21
  %22 = load i32, ptr @t, align 4, !tbaa !4
  %add18 = add nsw i32 %add17, %22
  ret i32 %add18
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.frameaddress.p0(i32 immarg) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #2

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr) #3

declare void @foo(ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef) local_unnamed_addr #4

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #5

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #3 = { nounwind }
attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #5 = { nofree nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git 79880371396d6e486bf6bacd6c4087ebdac591f8)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
