; Test output of setjmp/longjmp with 20 global variable sum(regsiter pressure).
; RUN: clang -mbackchain -o %t %s
; RUN: %t | FileCheck %s

; ModuleID = 'builtin-setjmp-spills-int-01.c'
source_filename = "builtin-setjmp-spills-int-01.c"
target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-unknown-linux-gnu"

@buf = dso_local global [10 x ptr] zeroinitializer, align 8
@t = dso_local local_unnamed_addr global i32 0, align 4
@s = dso_local local_unnamed_addr global i32 0, align 4
@r = dso_local local_unnamed_addr global i32 0, align 4
@q = dso_local local_unnamed_addr global i32 0, align 4
@p = dso_local local_unnamed_addr global i32 0, align 4
@o = dso_local local_unnamed_addr global i32 0, align 4
@n = dso_local local_unnamed_addr global i32 0, align 4
@m = dso_local local_unnamed_addr global i32 0, align 4
@l = dso_local local_unnamed_addr global i32 0, align 4
@k = dso_local local_unnamed_addr global i32 0, align 4
@j = dso_local local_unnamed_addr global i32 0, align 4
@i = dso_local local_unnamed_addr global i32 0, align 4
@h = dso_local local_unnamed_addr global i32 0, align 4
@g = dso_local local_unnamed_addr global i32 0, align 4
@f = dso_local local_unnamed_addr global i32 0, align 4
@e = dso_local local_unnamed_addr global i32 0, align 4
@d = dso_local local_unnamed_addr global i32 0, align 4
@c = dso_local local_unnamed_addr global i32 0, align 4
@b = dso_local local_unnamed_addr global i32 0, align 4
@a = dso_local local_unnamed_addr global i32 0, align 4
@.str.3 = private unnamed_addr constant [8 x i8] c"a = %d\0A\00", align 2
@.str.4 = private unnamed_addr constant [8 x i8] c"b = %d\0A\00", align 2
@.str.5 = private unnamed_addr constant [8 x i8] c"c = %d\0A\00", align 2
@.str.6 = private unnamed_addr constant [8 x i8] c"d = %d\0A\00", align 2
@.str.7 = private unnamed_addr constant [8 x i8] c"e = %d\0A\00", align 2
@.str.8 = private unnamed_addr constant [8 x i8] c"f = %d\0A\00", align 2
@.str.9 = private unnamed_addr constant [8 x i8] c"g = %d\0A\00", align 2
@.str.10 = private unnamed_addr constant [8 x i8] c"h = %d\0A\00", align 2
@.str.11 = private unnamed_addr constant [8 x i8] c"i = %d\0A\00", align 2
@.str.12 = private unnamed_addr constant [8 x i8] c"j = %d\0A\00", align 2
@.str.13 = private unnamed_addr constant [8 x i8] c"k = %d\0A\00", align 2
@.str.14 = private unnamed_addr constant [8 x i8] c"l = %d\0A\00", align 2
@.str.15 = private unnamed_addr constant [8 x i8] c"m = %d\0A\00", align 2
@.str.16 = private unnamed_addr constant [8 x i8] c"n = %d\0A\00", align 2
@.str.17 = private unnamed_addr constant [8 x i8] c"o = %d\0A\00", align 2
@.str.18 = private unnamed_addr constant [8 x i8] c"p = %d\0A\00", align 2
@.str.19 = private unnamed_addr constant [8 x i8] c"q = %d\0A\00", align 2
@.str.20 = private unnamed_addr constant [8 x i8] c"r = %d\0A\00", align 2
@.str.21 = private unnamed_addr constant [8 x i8] c"s = %d\0A\00", align 2
@.str.22 = private unnamed_addr constant [8 x i8] c"t = %d\0A\00", align 2
@.str.23 = private unnamed_addr constant [11 x i8] c"val is %d\0A\00", align 2
@str = private unnamed_addr constant [8 x i8] c"In func\00", align 1
@str.24 = private unnamed_addr constant [41 x i8] c"Second time through, checking variables:\00", align 1
@str.25 = private unnamed_addr constant [40 x i8] c"First time through, all variables are 1\00", align 1

; Function Attrs: noinline noreturn nounwind
define dso_local void @func1() local_unnamed_addr #0 {
entry:
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf)
  unreachable
}

; Function Attrs: nofree nounwind
declare noundef signext i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #1

; Function Attrs: noreturn nounwind
declare void @llvm.eh.sjlj.longjmp(ptr) #2

; Function Attrs: noinline nounwind
define dso_local signext i32 @func() local_unnamed_addr #3 {
entry:
; CHECK: Second time through, checking variables:
; CHECK: a = 1
; CHECK: b = 1
; CHECK: c = 1
; CHECK: d = 1
; CHECK: e = 1
; CHECK: f = 1
; CHECK: g = 1
; CHECK: h = 1
; CHECK: i = 1
; CHECK: j = 1
; CHECK: k = 1
; CHECK: l = 1
; CHECK: m = 1
; CHECK: n = 1
; CHECK: o = 1
; CHECK: p = 1
; CHECK: q = 1
; CHECK: r = 1
; CHECK: s = 1
; CHECK: t = 1

  %0 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf)
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

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
  %puts40 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.25)
  tail call void @func1()
  unreachable

if.else:                                          ; preds = %entry
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.24)
  %1 = load i32, ptr @a, align 4, !tbaa !4
  %call2 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef signext %1)
  %2 = load i32, ptr @b, align 4, !tbaa !4
  %call3 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef signext %2)
  %3 = load i32, ptr @c, align 4, !tbaa !4
  %call4 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef signext %3)
  %4 = load i32, ptr @d, align 4, !tbaa !4
  %call5 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %4)
  %5 = load i32, ptr @e, align 4, !tbaa !4
  %call6 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %5)
  %6 = load i32, ptr @f, align 4, !tbaa !4
  %call7 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8, i32 noundef signext %6)
  %7 = load i32, ptr @g, align 4, !tbaa !4
  %call8 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.9, i32 noundef signext %7)
  %8 = load i32, ptr @h, align 4, !tbaa !4
  %call9 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.10, i32 noundef signext %8)
  %9 = load i32, ptr @i, align 4, !tbaa !4
  %call10 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef signext %9)
  %10 = load i32, ptr @j, align 4, !tbaa !4
  %call11 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.12, i32 noundef signext %10)
  %11 = load i32, ptr @k, align 4, !tbaa !4
  %call12 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.13, i32 noundef signext %11)
  %12 = load i32, ptr @l, align 4, !tbaa !4
  %call13 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.14, i32 noundef signext %12)
  %13 = load i32, ptr @m, align 4, !tbaa !4
  %call14 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.15, i32 noundef signext %13)
  %14 = load i32, ptr @n, align 4, !tbaa !4
  %call15 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.16, i32 noundef signext %14)
  %15 = load i32, ptr @o, align 4, !tbaa !4
  %call16 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.17, i32 noundef signext %15)
  %16 = load i32, ptr @p, align 4, !tbaa !4
  %call17 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.18, i32 noundef signext %16)
  %17 = load i32, ptr @q, align 4, !tbaa !4
  %call18 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.19, i32 noundef signext %17)
  %18 = load i32, ptr @r, align 4, !tbaa !4
  %call19 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.20, i32 noundef signext %18)
  %19 = load i32, ptr @s, align 4, !tbaa !4
  %call20 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.21, i32 noundef signext %19)
  %20 = load i32, ptr @t, align 4, !tbaa !4
  %call21 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.22, i32 noundef signext %20)
  %21 = load i32, ptr @a, align 4, !tbaa !4
  %22 = load i32, ptr @b, align 4, !tbaa !4
  %add = add nsw i32 %22, %21
  %23 = load i32, ptr @c, align 4, !tbaa !4
  %add22 = add nsw i32 %add, %23
  %24 = load i32, ptr @d, align 4, !tbaa !4
  %add23 = add nsw i32 %add22, %24
  %25 = load i32, ptr @e, align 4, !tbaa !4
  %add24 = add nsw i32 %add23, %25
  %26 = load i32, ptr @f, align 4, !tbaa !4
  %add25 = add nsw i32 %add24, %26
  %27 = load i32, ptr @g, align 4, !tbaa !4
  %add26 = add nsw i32 %add25, %27
  %28 = load i32, ptr @h, align 4, !tbaa !4
  %add27 = add nsw i32 %add26, %28
  %29 = load i32, ptr @i, align 4, !tbaa !4
  %add28 = add nsw i32 %add27, %29
  %30 = load i32, ptr @j, align 4, !tbaa !4
  %add29 = add nsw i32 %add28, %30
  %31 = load i32, ptr @k, align 4, !tbaa !4
  %add30 = add nsw i32 %add29, %31
  %32 = load i32, ptr @l, align 4, !tbaa !4
  %add31 = add nsw i32 %add30, %32
  %33 = load i32, ptr @m, align 4, !tbaa !4
  %add32 = add nsw i32 %add31, %33
  %34 = load i32, ptr @n, align 4, !tbaa !4
  %add33 = add nsw i32 %add32, %34
  %35 = load i32, ptr @o, align 4, !tbaa !4
  %add34 = add nsw i32 %add33, %35
  %36 = load i32, ptr @p, align 4, !tbaa !4
  %add35 = add nsw i32 %add34, %36
  %37 = load i32, ptr @q, align 4, !tbaa !4
  %add36 = add nsw i32 %add35, %37
  %38 = load i32, ptr @r, align 4, !tbaa !4
  %add37 = add nsw i32 %add36, %38
  %39 = load i32, ptr @s, align 4, !tbaa !4
  %add38 = add nsw i32 %add37, %39
  %40 = load i32, ptr @t, align 4, !tbaa !4
  %add39 = add nsw i32 %add38, %40
  ret i32 %add39
}

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr) #4

; Function Attrs: nounwind
define dso_local noundef signext i32 @main() local_unnamed_addr #5 {
entry:
; CHECK: val is 20
  %call = tail call signext i32 @func()
  %call1 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, i32 noundef signext %call)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #6

attributes #0 = { noinline noreturn nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { nofree nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #2 = { noreturn nounwind }
attributes #3 = { noinline nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #4 = { nounwind }
attributes #5 = { nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #6 = { nofree nounwind }

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
