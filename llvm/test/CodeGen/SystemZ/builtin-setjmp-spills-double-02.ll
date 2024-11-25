; Simulate register pressure  around setjmp call for double precision arguments.
; Test assembly of funtion call foo in func() in setjmp if and else part.
; extern foo has 20 argument pointer to double precision.
; Test setjmp  store jmp_buf.
; Return address in slot 2.
; Stack Pointer in slot 4.
; Clobber %r6-%r15, %f8-%f15.

; RUN: llc -O2 < %s | FileCheck %s

@buf = dso_local global [10 x ptr] zeroinitializer, align 8
@t = dso_local global double 0.000000e+00, align 8
@s = dso_local global double 0.000000e+00, align 8
@r = dso_local global double 0.000000e+00, align 8
@q = dso_local global double 0.000000e+00, align 8
@p = dso_local global double 0.000000e+00, align 8
@o = dso_local global double 0.000000e+00, align 8
@n = dso_local global double 0.000000e+00, align 8
@m = dso_local global double 0.000000e+00, align 8
@l = dso_local global double 0.000000e+00, align 8
@k = dso_local global double 0.000000e+00, align 8
@j = dso_local global double 0.000000e+00, align 8
@i = dso_local global double 0.000000e+00, align 8
@h = dso_local global double 0.000000e+00, align 8
@g = dso_local global double 0.000000e+00, align 8
@f = dso_local global double 0.000000e+00, align 8
@e = dso_local global double 0.000000e+00, align 8
@d = dso_local global double 0.000000e+00, align 8
@c = dso_local global double 0.000000e+00, align 8
@b = dso_local global double 0.000000e+00, align 8
@a = dso_local global double 0.000000e+00, align 8
@str = private unnamed_addr constant [40 x i8] c"First time through, all variables are 1\00", align 1

; Function Attrs: nounwind
define dso_local double @func() local_unnamed_addr #0 {
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
; CHECK: larl    %r0, .LBB0_2
; CHECK: stg     %r0, 8(%r1)
; CHECK: stg     %r15, 24(%r1)

  %0 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf)
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store double 1.000000e+00, ptr @t, align 8, !tbaa !4
  store double 1.000000e+00, ptr @s, align 8, !tbaa !4
  store double 1.000000e+00, ptr @r, align 8, !tbaa !4
  store double 1.000000e+00, ptr @q, align 8, !tbaa !4
  store double 1.000000e+00, ptr @p, align 8, !tbaa !4
  store double 1.000000e+00, ptr @o, align 8, !tbaa !4
  store double 1.000000e+00, ptr @n, align 8, !tbaa !4
  store double 1.000000e+00, ptr @m, align 8, !tbaa !4
  store double 1.000000e+00, ptr @l, align 8, !tbaa !4
  store double 1.000000e+00, ptr @k, align 8, !tbaa !4
  store double 1.000000e+00, ptr @j, align 8, !tbaa !4
  store double 1.000000e+00, ptr @i, align 8, !tbaa !4
  store double 1.000000e+00, ptr @h, align 8, !tbaa !4
  store double 1.000000e+00, ptr @g, align 8, !tbaa !4
  store double 1.000000e+00, ptr @f, align 8, !tbaa !4
  store double 1.000000e+00, ptr @e, align 8, !tbaa !4
  store double 1.000000e+00, ptr @d, align 8, !tbaa !4
  store double 1.000000e+00, ptr @c, align 8, !tbaa !4
  store double 1.000000e+00, ptr @b, align 8, !tbaa !4
  store double 1.000000e+00, ptr @a, align 8, !tbaa !4
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  tail call void @foo(ptr noundef nonnull @a, ptr noundef nonnull @b, ptr noundef nonnull @c, ptr noundef nonnull @d, ptr noundef nonnull @e, ptr noundef nonnull @f, ptr noundef nonnull @g, ptr noundef nonnull @h, ptr noundef nonnull @i, ptr noundef nonnull @j, ptr noundef nonnull @k, ptr noundef nonnull @l, ptr noundef nonnull @m, ptr noundef nonnull @n, ptr noundef nonnull @o, ptr noundef nonnull @p, ptr noundef nonnull @q, ptr noundef nonnull @r, ptr noundef nonnull @s, ptr noundef nonnull @t) #1
  %1 = load double, ptr @a, align 8, !tbaa !4
  %2 = load double, ptr @b, align 8, !tbaa !4
  %add = fadd double %1, %2
  %3 = load double, ptr @c, align 8, !tbaa !4
  %add1 = fadd double %add, %3
  %4 = load double, ptr @d, align 8, !tbaa !4
  %add2 = fadd double %add1, %4
  %5 = load double, ptr @e, align 8, !tbaa !4
  %add3 = fadd double %add2, %5
  %6 = load double, ptr @f, align 8, !tbaa !4
  %add4 = fadd double %add3, %6
  %7 = load double, ptr @g, align 8, !tbaa !4
  %add5 = fadd double %add4, %7
  %8 = load double, ptr @h, align 8, !tbaa !4
  %add6 = fadd double %add5, %8
  %9 = load double, ptr @i, align 8, !tbaa !4
  %add7 = fadd double %add6, %9
  %10 = load double, ptr @j, align 8, !tbaa !4
  %add8 = fadd double %add7, %10
  %11 = load double, ptr @k, align 8, !tbaa !4
  %add9 = fadd double %add8, %11
  %12 = load double, ptr @l, align 8, !tbaa !4
  %add10 = fadd double %add9, %12
  %13 = load double, ptr @m, align 8, !tbaa !4
  %add11 = fadd double %add10, %13
  %14 = load double, ptr @n, align 8, !tbaa !4
  %add12 = fadd double %add11, %14
  %15 = load double, ptr @o, align 8, !tbaa !4
  %add13 = fadd double %add12, %15
  %16 = load double, ptr @p, align 8, !tbaa !4
  %add14 = fadd double %add13, %16
  %17 = load double, ptr @q, align 8, !tbaa !4
  %add15 = fadd double %add14, %17
  %18 = load double, ptr @r, align 8, !tbaa !4
  %add16 = fadd double %add15, %18
  %19 = load double, ptr @s, align 8, !tbaa !4
  %add17 = fadd double %add16, %19
  %20 = load double, ptr @t, align 8, !tbaa !4
  %add18 = fadd double %add17, %20
  ret double %add18
}

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr) #1

declare void @foo(ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef) local_unnamed_addr #2

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #3

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { nounwind }
attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #3 = { nofree nounwind }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!4 = !{!5, !5, i64 0}
!5 = !{!"double", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
