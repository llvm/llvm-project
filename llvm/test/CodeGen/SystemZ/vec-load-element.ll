; Test that selection of Vector Load Element instructions work in the presence of prefetches.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; CHECK-LABEL: .LBB0_1:
; CHECK-NOT: l %r
; CHECK-NOT: vlvgf
; CHECK: pfd
; CHECK: vlef

%type0 = type { i32, [400 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
@Mem = external global [150 x %type0], align 4

define void @fun() local_unnamed_addr #0 {
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next.3, %vector.body ]
  %vec.phi = phi <4 x i32> [ zeroinitializer, %entry ], [ %57, %vector.body ]
  %0 = or i64 %index, 2
  %1 = or i64 %index, 3
  %2 = getelementptr inbounds [150 x %type0], ptr @Mem, i64 0, i64 0, i32 3
  %3 = getelementptr inbounds [150 x %type0], ptr @Mem, i64 0, i64 %0, i32 3
  %4 = getelementptr inbounds [150 x %type0], ptr @Mem, i64 0, i64 %1, i32 3
  %5 = load i32, ptr null, align 4
  %6 = load i32, ptr %3, align 4
  %7 = load i32, ptr %4, align 4
  %8 = insertelement <4 x i32> undef, i32 %5, i32 0
  %9 = insertelement <4 x i32> %8, i32 0, i32 1
  %10 = insertelement <4 x i32> %9, i32 %6, i32 2
  %11 = insertelement <4 x i32> %10, i32 %7, i32 3
  %12 = add nsw <4 x i32> %11, %vec.phi
  %13 = or i64 %index, 7
  %14 = getelementptr inbounds [150 x %type0], ptr @Mem, i64 0, i64 undef, i32 3
  %15 = getelementptr inbounds [150 x %type0], ptr @Mem, i64 0, i64 0, i32 3
  %16 = getelementptr inbounds [150 x %type0], ptr @Mem, i64 0, i64 %13, i32 3
  %17 = load i32, ptr %14, align 4
  %18 = load i32, ptr undef, align 4
  %19 = load i32, ptr %15, align 4
  %20 = load i32, ptr %16, align 4
  %21 = insertelement <4 x i32> undef, i32 %17, i32 0
  %22 = insertelement <4 x i32> %21, i32 %18, i32 1
  %23 = insertelement <4 x i32> %22, i32 %19, i32 2
  %24 = insertelement <4 x i32> %23, i32 %20, i32 3
  %25 = add nsw <4 x i32> %24, %12
  %26 = or i64 %index, 9
  %27 = or i64 %index, 10
  %28 = or i64 %index, 11
  %29 = getelementptr inbounds [150 x %type0], ptr @Mem, i64 0, i64 undef, i32 3
  %30 = getelementptr inbounds [150 x %type0], ptr @Mem, i64 0, i64 %26, i32 3
  %31 = getelementptr inbounds [150 x %type0], ptr @Mem, i64 0, i64 %27, i32 3
  %32 = getelementptr inbounds [150 x %type0], ptr @Mem, i64 0, i64 %28, i32 3
  %33 = load i32, ptr %29, align 4
  %34 = load i32, ptr %30, align 4
  %35 = load i32, ptr %31, align 4
  %36 = load i32, ptr %32, align 4
  %37 = insertelement <4 x i32> undef, i32 %33, i32 0
  %38 = insertelement <4 x i32> %37, i32 %34, i32 1
  %39 = insertelement <4 x i32> %38, i32 %35, i32 2
  %40 = insertelement <4 x i32> %39, i32 %36, i32 3
  %41 = add nsw <4 x i32> %40, %25
  %42 = or i64 %index, 13
  %43 = or i64 %index, 14
  %44 = or i64 %index, 15
  %45 = getelementptr inbounds [150 x %type0], ptr @Mem, i64 0, i64 undef, i32 3
  %46 = getelementptr inbounds [150 x %type0], ptr @Mem, i64 0, i64 %42, i32 3
  %47 = getelementptr inbounds [150 x %type0], ptr @Mem, i64 0, i64 %43, i32 3
  %48 = getelementptr inbounds [150 x %type0], ptr @Mem, i64 0, i64 %44, i32 3
  %49 = load i32, ptr %45, align 4
  %50 = load i32, ptr %46, align 4
  %51 = load i32, ptr %47, align 4
  %52 = load i32, ptr %48, align 4
  %53 = insertelement <4 x i32> undef, i32 %49, i32 0
  %54 = insertelement <4 x i32> %53, i32 %50, i32 1
  %55 = insertelement <4 x i32> %54, i32 %51, i32 2
  %56 = insertelement <4 x i32> %55, i32 %52, i32 3
  %57 = add nsw <4 x i32> %56, %41
  %index.next.3 = add i64 %index, 16
  br i1 false, label %middle.block.unr-lcssa, label %vector.body

middle.block.unr-lcssa:                           ; preds = %vector.body
  %rdx.shuf = shufflevector <4 x i32> %57, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  unreachable
}

