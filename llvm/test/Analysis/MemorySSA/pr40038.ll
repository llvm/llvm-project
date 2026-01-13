; REQUIRES: asserts
; RUN: opt -S -mtriple=systemz-unknown -mcpu=z13  -O3 -verify-memoryssa  < %s | FileCheck %s

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-ibm-linux"

@g_80 = external dso_local global i32, align 4
@g_1683 = external dso_local global i32, align 4
@0 = internal global [7 x i8] c"\00\EE\00\00\EE\00\00", align 2

; Function Attrs: nounwind
; CHECK-LABEL: @main
define dso_local void @main() {
bb:
  call void @func_1()
  unreachable
}

; Function Attrs: nounwind
define dso_local void @func_1() {
bb:
  call void @func_2()
  unreachable
}

; Function Attrs: nounwind
define dso_local void @func_2() {
bb:
  %tmp = alloca i32, align 4
  store i32 0, ptr @g_80, align 4, !tbaa !1
  br label %bb1

bb1:                                              ; preds = %bb15, %bb
  %tmp2 = load i32, ptr @g_80, align 4, !tbaa !1
  %tmp3 = icmp sle i32 %tmp2, 6
  br i1 %tmp3, label %bb4, label %bb18

bb4:                                              ; preds = %bb1
  %tmp5 = load i32, ptr @g_1683, align 4, !tbaa !1
  %tmp6 = sext i32 %tmp5 to i64
  %tmp7 = getelementptr inbounds [7 x i8], ptr @0, i64 0, i64 %tmp6
  %tmp8 = load i8, ptr %tmp7, align 1, !tbaa !5
  %tmp9 = icmp ne i8 %tmp8, 0
  br i1 %tmp9, label %bb10, label %bb11

bb10:                                             ; preds = %bb4
  store i32 82, ptr %tmp, align 4
  br label %bb12

bb11:                                             ; preds = %bb4
  store i32 0, ptr %tmp, align 4
  br label %bb12

bb12:                                             ; preds = %bb11, %bb10
  %tmp13 = load i32, ptr %tmp, align 4
  %tmp14 = icmp ult i32 %tmp13, 1
  br i1 %tmp14, label %bb15, label %bb18

bb15:                                             ; preds = %bb12
  %tmp16 = load i32, ptr @g_80, align 4, !tbaa !1
  %tmp17 = add nsw i32 %tmp16, 1
  store i32 %tmp17, ptr @g_80, align 4, !tbaa !1
  br label %bb1

bb18:                                             ; preds = %bb12, %bb1
  call void @llvm.trap()
  unreachable
}

; Function Attrs: cold noreturn nounwind
declare void @llvm.trap()

!llvm.ident = !{!0}

!0 = !{!"clang version 8.0.0 (http://llvm.org/git/clang.git a674a04e68bcf09f9a0423f3f589589596bc01a6) (http://llvm.org/git/llvm.git 1fe1ffe00e034128d1c5504254fdd4742f48bb9a)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!3, !3, i64 0}
