; -mbackchain
; Non-volatile local malloc'd variable being modified between setjmp and longjmp call.
; size of malloc is 25 and array variable is not kept on stack.
; This test is with optimization -O2, modified value persists.
; For size of malloc 20, all array values are stored to stack, and modified
; values does not persist.
; RUN: clang -mbackchain -O2 -o %t %s
; RUN: %t | FileCheck %s

; CHECK: Returned from func4
; CHECK: arr: 0
; CHECK: arr: 2
; CHECK: arr: 6
; CHECK: arr: 12
; CHECK: arr: 20
; CHECK: arr: 30
; CHECK: arr: 42
; CHECK: arr: 56
; CHECK: arr: 72
; CHECK: arr: 90
; CHECK: arr: 110
; CHECK: arr: 132
; CHECK: arr: 156
; CHECK: arr: 182
; CHECK: arr: 210
; CHECK: arr: 240
; CHECK: arr: 272
; CHECK: arr: 306
; CHECK: arr: 342
; CHECK: arr: 380
; CHECK: arr: 420
; CHECK: arr: 462
; CHECK: arr: 506
; CHECK: arr: 552
; CHECK: arr: 600
; CHECK: In func3
; CHECK: Returned from func3
; CHECK: arr: 0
; CHECK: arr: 3
; CHECK: arr: 14
; CHECK: arr: 39
; CHECK: arr: 84
; CHECK: arr: 155
; CHECK: arr: 258
; CHECK: arr: 399
; CHECK: arr: 584
; CHECK: arr: 819
; CHECK: arr: 1110
; CHECK: arr: 1463
; CHECK: arr: 1884
; CHECK: arr: 2379
; CHECK: arr: 2954
; CHECK: arr: 3615
; CHECK: arr: 4368
; CHECK: arr: 5219
; CHECK: arr: 6174
; CHECK: arr: 7239
; CHECK: arr: 8420
; CHECK: arr: 9723
; CHECK: arr: 11154
; CHECK: arr: 12719
; CHECK: arr: 14424

; ModuleID = 'builtin-setjmp-longjmp-malloc-local-03.c'
source_filename = "builtin-setjmp-longjmp-malloc-local-03.c"
target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-unknown-linux-gnu"

@buf3 = dso_local global [10 x ptr] zeroinitializer, align 8
@buf2 = dso_local global [10 x ptr] zeroinitializer, align 8
@buf1 = dso_local global [10 x ptr] zeroinitializer, align 8
@.str.6 = private unnamed_addr constant [9 x i8] c"arr: %d\0A\00", align 2
@str = private unnamed_addr constant [9 x i8] c"In func4\00", align 1
@str.10 = private unnamed_addr constant [9 x i8] c"In func3\00", align 1
@str.11 = private unnamed_addr constant [9 x i8] c"In func2\00", align 1
@str.12 = private unnamed_addr constant [20 x i8] c"Returned from func3\00", align 1
@str.13 = private unnamed_addr constant [32 x i8] c"First __builtin_setjmp in func1\00", align 1
@str.14 = private unnamed_addr constant [20 x i8] c"Returned from func4\00", align 1
@str.15 = private unnamed_addr constant [33 x i8] c"Second __builtin_setjmp in func1\00", align 1
@str.16 = private unnamed_addr constant [44 x i8] c"In main, after __builtin_longjmp from func1\00", align 1
@str.17 = private unnamed_addr constant [20 x i8] c"In main, first time\00", align 1

; Function Attrs: noinline noreturn nounwind
define dso_local void @func4() local_unnamed_addr #0 {
entry:
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf3)
  unreachable
}

; Function Attrs: nofree nounwind
declare noundef signext i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #1

; Function Attrs: noreturn nounwind
declare void @llvm.eh.sjlj.longjmp(ptr) #2

; Function Attrs: noinline noreturn nounwind
define dso_local void @func3() local_unnamed_addr #0 {
entry:
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.10)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf2)
  unreachable
}

; Function Attrs: noinline noreturn nounwind
define dso_local void @func2() local_unnamed_addr #0 {
entry:
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.11)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf1)
  unreachable
}

; Function Attrs: noreturn nounwind
define dso_local noundef signext i32 @func1() local_unnamed_addr #3 {
entry:
  %call = tail call noalias dereferenceable_or_null(100) ptr @malloc(i64 noundef 100) #9
  store i32 0, ptr %call, align 4, !tbaa !4
  %arrayidx.1 = getelementptr inbounds i8, ptr %call, i64 4
  store i32 2, ptr %arrayidx.1, align 4, !tbaa !4
  %arrayidx.2 = getelementptr inbounds i8, ptr %call, i64 8
  store i32 6, ptr %arrayidx.2, align 4, !tbaa !4
  %arrayidx.3 = getelementptr inbounds i8, ptr %call, i64 12
  store i32 12, ptr %arrayidx.3, align 4, !tbaa !4
  %arrayidx.4 = getelementptr inbounds i8, ptr %call, i64 16
  store i32 20, ptr %arrayidx.4, align 4, !tbaa !4
  %arrayidx.5 = getelementptr inbounds i8, ptr %call, i64 20
  store i32 30, ptr %arrayidx.5, align 4, !tbaa !4
  %arrayidx.6 = getelementptr inbounds i8, ptr %call, i64 24
  store i32 42, ptr %arrayidx.6, align 4, !tbaa !4
  %arrayidx.7 = getelementptr inbounds i8, ptr %call, i64 28
  store i32 56, ptr %arrayidx.7, align 4, !tbaa !4
  %arrayidx.8 = getelementptr inbounds i8, ptr %call, i64 32
  store i32 72, ptr %arrayidx.8, align 4, !tbaa !4
  %arrayidx.9 = getelementptr inbounds i8, ptr %call, i64 36
  store i32 90, ptr %arrayidx.9, align 4, !tbaa !4
  %arrayidx.10 = getelementptr inbounds i8, ptr %call, i64 40
  store i32 110, ptr %arrayidx.10, align 4, !tbaa !4
  %arrayidx.11 = getelementptr inbounds i8, ptr %call, i64 44
  store i32 132, ptr %arrayidx.11, align 4, !tbaa !4
  %arrayidx.12 = getelementptr inbounds i8, ptr %call, i64 48
  store i32 156, ptr %arrayidx.12, align 4, !tbaa !4
  %arrayidx.13 = getelementptr inbounds i8, ptr %call, i64 52
  store i32 182, ptr %arrayidx.13, align 4, !tbaa !4
  %arrayidx.14 = getelementptr inbounds i8, ptr %call, i64 56
  store i32 210, ptr %arrayidx.14, align 4, !tbaa !4
  %arrayidx.15 = getelementptr inbounds i8, ptr %call, i64 60
  store i32 240, ptr %arrayidx.15, align 4, !tbaa !4
  %arrayidx.16 = getelementptr inbounds i8, ptr %call, i64 64
  store i32 272, ptr %arrayidx.16, align 4, !tbaa !4
  %arrayidx.17 = getelementptr inbounds i8, ptr %call, i64 68
  store i32 306, ptr %arrayidx.17, align 4, !tbaa !4
  %arrayidx.18 = getelementptr inbounds i8, ptr %call, i64 72
  store i32 342, ptr %arrayidx.18, align 4, !tbaa !4
  %arrayidx.19 = getelementptr inbounds i8, ptr %call, i64 76
  store i32 380, ptr %arrayidx.19, align 4, !tbaa !4
  %arrayidx.20 = getelementptr inbounds i8, ptr %call, i64 80
  store i32 420, ptr %arrayidx.20, align 4, !tbaa !4
  %arrayidx.21 = getelementptr inbounds i8, ptr %call, i64 84
  store i32 462, ptr %arrayidx.21, align 4, !tbaa !4
  %arrayidx.22 = getelementptr inbounds i8, ptr %call, i64 88
  store i32 506, ptr %arrayidx.22, align 4, !tbaa !4
  %arrayidx.23 = getelementptr inbounds i8, ptr %call, i64 92
  store i32 552, ptr %arrayidx.23, align 4, !tbaa !4
  %arrayidx.24 = getelementptr inbounds i8, ptr %call, i64 96
  store i32 600, ptr %arrayidx.24, align 4, !tbaa !4
  %0 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf2)
  %cmp3 = icmp eq i32 %0, 0
  br i1 %cmp3, label %if.then, label %if.else39

if.then:                                          ; preds = %entry
  %puts79 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.13)
  %1 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf3)
  %cmp6 = icmp eq i32 %1, 0
  br i1 %cmp6, label %if.then8, label %if.else

if.then8:                                         ; preds = %if.then
  %puts84 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.15)
  tail call void @func4()
  unreachable

if.else:                                          ; preds = %if.then
  %puts80 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.14)
  %2 = load i32, ptr %call, align 4, !tbaa !4
  %call19 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %2)
  %3 = load i32, ptr %arrayidx.1, align 4, !tbaa !4
  %call19.1 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %3)
  %4 = load i32, ptr %arrayidx.2, align 4, !tbaa !4
  %call19.2 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %4)
  %5 = load i32, ptr %arrayidx.3, align 4, !tbaa !4
  %call19.3 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %5)
  %6 = load i32, ptr %arrayidx.4, align 4, !tbaa !4
  %call19.4 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %6)
  %7 = load i32, ptr %arrayidx.5, align 4, !tbaa !4
  %call19.5 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %7)
  %8 = load i32, ptr %arrayidx.6, align 4, !tbaa !4
  %call19.6 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %8)
  %9 = load i32, ptr %arrayidx.7, align 4, !tbaa !4
  %call19.7 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %9)
  %10 = load i32, ptr %arrayidx.8, align 4, !tbaa !4
  %call19.8 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %10)
  %11 = load i32, ptr %arrayidx.9, align 4, !tbaa !4
  %call19.9 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %11)
  %12 = load i32, ptr %arrayidx.10, align 4, !tbaa !4
  %call19.10 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %12)
  %13 = load i32, ptr %arrayidx.11, align 4, !tbaa !4
  %call19.11 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %13)
  %14 = load i32, ptr %arrayidx.12, align 4, !tbaa !4
  %call19.12 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %14)
  %15 = load i32, ptr %arrayidx.13, align 4, !tbaa !4
  %call19.13 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %15)
  %16 = load i32, ptr %arrayidx.14, align 4, !tbaa !4
  %call19.14 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %16)
  %17 = load i32, ptr %arrayidx.15, align 4, !tbaa !4
  %call19.15 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %17)
  %18 = load i32, ptr %arrayidx.16, align 4, !tbaa !4
  %call19.16 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %18)
  %19 = load i32, ptr %arrayidx.17, align 4, !tbaa !4
  %call19.17 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %19)
  %20 = load i32, ptr %arrayidx.18, align 4, !tbaa !4
  %call19.18 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %20)
  %21 = load i32, ptr %arrayidx.19, align 4, !tbaa !4
  %call19.19 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %21)
  %22 = load i32, ptr %arrayidx.20, align 4, !tbaa !4
  %call19.20 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %22)
  %23 = load i32, ptr %arrayidx.21, align 4, !tbaa !4
  %call19.21 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %23)
  %24 = load i32, ptr %arrayidx.22, align 4, !tbaa !4
  %call19.22 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %24)
  %25 = load i32, ptr %arrayidx.23, align 4, !tbaa !4
  %call19.23 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %25)
  %26 = load i32, ptr %arrayidx.24, align 4, !tbaa !4
  %call19.24 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %26)
  br label %for.body28

for.cond.cleanup27:                               ; preds = %for.body28
  tail call void @func3()
  unreachable

for.body28:                                       ; preds = %for.body28, %if.else
  %indvars.iv = phi i64 [ 0, %if.else ], [ %indvars.iv.next.4, %for.body28 ]
  %indvars96 = trunc i64 %indvars.iv to i32
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %indvars = trunc i64 %indvars.iv.next to i32
  %mul3081 = mul nuw nsw i32 %indvars, %indvars96
  %add3283 = add nuw nsw i32 %mul3081, 1
  %add33 = mul nuw nsw i32 %add3283, %indvars96
  %arrayidx35 = getelementptr inbounds i32, ptr %call, i64 %indvars.iv
  store i32 %add33, ptr %arrayidx35, align 4, !tbaa !4
  %indvars96.1 = trunc i64 %indvars.iv.next to i32
  %indvars.iv.next.1 = add nuw nsw i64 %indvars.iv, 2
  %indvars.1 = trunc i64 %indvars.iv.next.1 to i32
  %mul3081.1 = mul nuw nsw i32 %indvars.1, %indvars96.1
  %add3283.1 = add nuw nsw i32 %mul3081.1, 1
  %add33.1 = mul nuw nsw i32 %add3283.1, %indvars96.1
  %arrayidx35.1 = getelementptr inbounds i32, ptr %call, i64 %indvars.iv.next
  store i32 %add33.1, ptr %arrayidx35.1, align 4, !tbaa !4
  %indvars96.2 = trunc i64 %indvars.iv.next.1 to i32
  %indvars.iv.next.2 = add nuw nsw i64 %indvars.iv, 3
  %indvars.2 = trunc i64 %indvars.iv.next.2 to i32
  %mul3081.2 = mul nuw nsw i32 %indvars.2, %indvars96.2
  %add3283.2 = add nuw nsw i32 %mul3081.2, 1
  %add33.2 = mul nuw nsw i32 %add3283.2, %indvars96.2
  %arrayidx35.2 = getelementptr inbounds i32, ptr %call, i64 %indvars.iv.next.1
  store i32 %add33.2, ptr %arrayidx35.2, align 4, !tbaa !4
  %indvars96.3 = trunc i64 %indvars.iv.next.2 to i32
  %indvars.iv.next.3 = add nuw nsw i64 %indvars.iv, 4
  %indvars.3 = trunc i64 %indvars.iv.next.3 to i32
  %mul3081.3 = mul nuw nsw i32 %indvars.3, %indvars96.3
  %add3283.3 = add nuw nsw i32 %mul3081.3, 1
  %add33.3 = mul nuw nsw i32 %add3283.3, %indvars96.3
  %arrayidx35.3 = getelementptr inbounds i32, ptr %call, i64 %indvars.iv.next.2
  store i32 %add33.3, ptr %arrayidx35.3, align 4, !tbaa !4
  %indvars96.4 = trunc i64 %indvars.iv.next.3 to i32
  %indvars.iv.next.4 = add nuw nsw i64 %indvars.iv, 5
  %indvars.4 = trunc i64 %indvars.iv.next.4 to i32
  %mul3081.4 = mul nuw nsw i32 %indvars.4, %indvars96.4
  %add3283.4 = add nuw nsw i32 %mul3081.4, 1
  %add33.4 = mul nuw nsw i32 %add3283.4, %indvars96.4
  %arrayidx35.4 = getelementptr inbounds i32, ptr %call, i64 %indvars.iv.next.3
  store i32 %add33.4, ptr %arrayidx35.4, align 4, !tbaa !4
  %exitcond.not.4 = icmp eq i64 %indvars.iv.next.4, 25
  br i1 %exitcond.not.4, label %for.cond.cleanup27, label %for.body28, !llvm.loop !8

if.else39:                                        ; preds = %entry
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.12)
  %27 = load i32, ptr %call, align 4, !tbaa !4
  %call49 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %27)
  %28 = load i32, ptr %arrayidx.1, align 4, !tbaa !4
  %call49.1 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %28)
  %29 = load i32, ptr %arrayidx.2, align 4, !tbaa !4
  %call49.2 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %29)
  %30 = load i32, ptr %arrayidx.3, align 4, !tbaa !4
  %call49.3 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %30)
  %31 = load i32, ptr %arrayidx.4, align 4, !tbaa !4
  %call49.4 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %31)
  %32 = load i32, ptr %arrayidx.5, align 4, !tbaa !4
  %call49.5 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %32)
  %33 = load i32, ptr %arrayidx.6, align 4, !tbaa !4
  %call49.6 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %33)
  %34 = load i32, ptr %arrayidx.7, align 4, !tbaa !4
  %call49.7 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %34)
  %35 = load i32, ptr %arrayidx.8, align 4, !tbaa !4
  %call49.8 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %35)
  %36 = load i32, ptr %arrayidx.9, align 4, !tbaa !4
  %call49.9 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %36)
  %37 = load i32, ptr %arrayidx.10, align 4, !tbaa !4
  %call49.10 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %37)
  %38 = load i32, ptr %arrayidx.11, align 4, !tbaa !4
  %call49.11 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %38)
  %39 = load i32, ptr %arrayidx.12, align 4, !tbaa !4
  %call49.12 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %39)
  %40 = load i32, ptr %arrayidx.13, align 4, !tbaa !4
  %call49.13 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %40)
  %41 = load i32, ptr %arrayidx.14, align 4, !tbaa !4
  %call49.14 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %41)
  %42 = load i32, ptr %arrayidx.15, align 4, !tbaa !4
  %call49.15 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %42)
  %43 = load i32, ptr %arrayidx.16, align 4, !tbaa !4
  %call49.16 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %43)
  %44 = load i32, ptr %arrayidx.17, align 4, !tbaa !4
  %call49.17 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %44)
  %45 = load i32, ptr %arrayidx.18, align 4, !tbaa !4
  %call49.18 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %45)
  %46 = load i32, ptr %arrayidx.19, align 4, !tbaa !4
  %call49.19 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %46)
  %47 = load i32, ptr %arrayidx.20, align 4, !tbaa !4
  %call49.20 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %47)
  %48 = load i32, ptr %arrayidx.21, align 4, !tbaa !4
  %call49.21 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %48)
  %49 = load i32, ptr %arrayidx.22, align 4, !tbaa !4
  %call49.22 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %49)
  %50 = load i32, ptr %arrayidx.23, align 4, !tbaa !4
  %call49.23 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %50)
  %51 = load i32, ptr %arrayidx.24, align 4, !tbaa !4
  %call49.24 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %51)
  tail call void @free(ptr noundef nonnull %call) #5
  tail call void @func2()
  unreachable
}

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #4

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr) #5

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr nocapture noundef) local_unnamed_addr #6

; Function Attrs: nounwind
define dso_local noundef signext i32 @main() local_unnamed_addr #7 {
entry:
  %0 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf1)
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %puts3 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.17)
  %call1 = tail call signext i32 @func1()
  unreachable

if.else:                                          ; preds = %entry
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.16)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #8

attributes #0 = { noinline noreturn nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { nofree nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #2 = { noreturn nounwind }
attributes #3 = { noreturn nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #4 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #5 = { nounwind }
attributes #6 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #7 = { nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #8 = { nofree nounwind }
attributes #9 = { nounwind allocsize(0) }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git a0433728375e658551506ce43b0848200fdd6e61)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.mustprogress"}
