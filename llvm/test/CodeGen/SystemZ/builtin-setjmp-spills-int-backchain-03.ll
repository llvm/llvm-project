; -mbackchain option.
; Simulate register pressure  around setjmp call for integer arguments and
; return sum of 20 vaiables. It also prints the variables.
; Test assembly of funtion call foo in func() in setjmp if and else part.
; extern foo has 20 argument pointer to int.
; Test setjmp  store jmp_buf.
; Return address in slot 2.
; Backchain value in slot 3.
; Stack Pointer in slot 4.
; Clobber %r6-%r15, %f8-%f15.

; RUN: llc -O2 < %s | FileCheck %s

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
@.str.2 = private unnamed_addr constant [8 x i8] c"a = %d\0A\00", align 2
@.str.3 = private unnamed_addr constant [8 x i8] c"b = %d\0A\00", align 2
@.str.4 = private unnamed_addr constant [8 x i8] c"c = %d\0A\00", align 2
@.str.5 = private unnamed_addr constant [8 x i8] c"d = %d\0A\00", align 2
@.str.6 = private unnamed_addr constant [8 x i8] c"e = %d\0A\00", align 2
@.str.7 = private unnamed_addr constant [8 x i8] c"f = %d\0A\00", align 2
@.str.8 = private unnamed_addr constant [8 x i8] c"g = %d\0A\00", align 2
@.str.9 = private unnamed_addr constant [8 x i8] c"h = %d\0A\00", align 2
@.str.10 = private unnamed_addr constant [8 x i8] c"i = %d\0A\00", align 2
@.str.11 = private unnamed_addr constant [8 x i8] c"j = %d\0A\00", align 2
@.str.12 = private unnamed_addr constant [8 x i8] c"k = %d\0A\00", align 2
@.str.13 = private unnamed_addr constant [8 x i8] c"l = %d\0A\00", align 2
@.str.14 = private unnamed_addr constant [8 x i8] c"m = %d\0A\00", align 2
@.str.15 = private unnamed_addr constant [8 x i8] c"n = %d\0A\00", align 2
@.str.16 = private unnamed_addr constant [8 x i8] c"o = %d\0A\00", align 2
@.str.17 = private unnamed_addr constant [8 x i8] c"p = %d\0A\00", align 2
@.str.18 = private unnamed_addr constant [8 x i8] c"q = %d\0A\00", align 2
@.str.19 = private unnamed_addr constant [8 x i8] c"r = %d\0A\00", align 2
@.str.20 = private unnamed_addr constant [8 x i8] c"s = %d\0A\00", align 2
@.str.21 = private unnamed_addr constant [8 x i8] c"t = %d\0A\00", align 2
@str = private unnamed_addr constant [41 x i8] c"Second time through, checking variables:\00", align 1
@str.22 = private unnamed_addr constant [40 x i8] c"First time through, all variables are 1\00", align 1

; Function Attrs: nounwind
define dso_local signext i32 @func() local_unnamed_addr #0 {
entry:
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
; CHECK: larl    %r1, buf
; CHECK: larl    %r0, .LBB0_3
; CHECK: stg     %r0, 8(%r1)
; CHECK: stg     %r15, 24(%r1)
; CHECK: lg      %r0, 0(%r15)
; CHECK: stg     %r0, 16(%r1)

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
  %puts40 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.22)
  tail call void @foo(ptr noundef nonnull @a, ptr noundef nonnull @b, ptr noundef nonnull @c, ptr noundef nonnull @d, ptr noundef nonnull @e, ptr noundef nonnull @f, ptr noundef nonnull @g, ptr noundef nonnull @h, ptr noundef nonnull @i, ptr noundef nonnull @j, ptr noundef nonnull @k, ptr noundef nonnull @l, ptr noundef nonnull @m, ptr noundef nonnull @n, ptr noundef nonnull @o, ptr noundef nonnull @p, ptr noundef nonnull @q, ptr noundef nonnull @r, ptr noundef nonnull @s, ptr noundef nonnull @t) #1
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void @foo(ptr noundef nonnull @a, ptr noundef nonnull @b, ptr noundef nonnull @c, ptr noundef nonnull @d, ptr noundef nonnull @e, ptr noundef nonnull @f, ptr noundef nonnull @g, ptr noundef nonnull @h, ptr noundef nonnull @i, ptr noundef nonnull @j, ptr noundef nonnull @k, ptr noundef nonnull @l, ptr noundef nonnull @m, ptr noundef nonnull @n, ptr noundef nonnull @o, ptr noundef nonnull @p, ptr noundef nonnull @q, ptr noundef nonnull @r, ptr noundef nonnull @s, ptr noundef nonnull @t) #1
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  %1 = load i32, ptr @a, align 4, !tbaa !4
  %call2 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef signext %1)
  %2 = load i32, ptr @b, align 4, !tbaa !4
  %call3 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef signext %2)
  %3 = load i32, ptr @c, align 4, !tbaa !4
  %call4 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef signext %3)
  %4 = load i32, ptr @d, align 4, !tbaa !4
  %call5 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef signext %4)
  %5 = load i32, ptr @e, align 4, !tbaa !4
  %call6 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %5)
  %6 = load i32, ptr @f, align 4, !tbaa !4
  %call7 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %6)
  %7 = load i32, ptr @g, align 4, !tbaa !4
  %call8 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8, i32 noundef signext %7)
  %8 = load i32, ptr @h, align 4, !tbaa !4
  %call9 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.9, i32 noundef signext %8)
  %9 = load i32, ptr @i, align 4, !tbaa !4
  %call10 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.10, i32 noundef signext %9)
  %10 = load i32, ptr @j, align 4, !tbaa !4
  %call11 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef signext %10)
  %11 = load i32, ptr @k, align 4, !tbaa !4
  %call12 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.12, i32 noundef signext %11)
  %12 = load i32, ptr @l, align 4, !tbaa !4
  %call13 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.13, i32 noundef signext %12)
  %13 = load i32, ptr @m, align 4, !tbaa !4
  %call14 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.14, i32 noundef signext %13)
  %14 = load i32, ptr @n, align 4, !tbaa !4
  %call15 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.15, i32 noundef signext %14)
  %15 = load i32, ptr @o, align 4, !tbaa !4
  %call16 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.16, i32 noundef signext %15)
  %16 = load i32, ptr @p, align 4, !tbaa !4
  %call17 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.17, i32 noundef signext %16)
  %17 = load i32, ptr @q, align 4, !tbaa !4
  %call18 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.18, i32 noundef signext %17)
  %18 = load i32, ptr @r, align 4, !tbaa !4
  %call19 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.19, i32 noundef signext %18)
  %19 = load i32, ptr @s, align 4, !tbaa !4
  %call20 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.20, i32 noundef signext %19)
  %20 = load i32, ptr @t, align 4, !tbaa !4
  %call21 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.21, i32 noundef signext %20)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
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
declare i32 @llvm.eh.sjlj.setjmp(ptr) #1

; Function Attrs: nofree nounwind
declare noundef signext i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #2

declare void @foo(ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef) local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #4

attributes #0 = { nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { nounwind }
attributes #2 = { nofree nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #3 = { "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #4 = { nofree nounwind }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
