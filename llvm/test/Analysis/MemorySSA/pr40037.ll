; REQUIRES: asserts
; RUN: opt -S -mtriple=systemz-unknown -mcpu=z13  -O3 -enable-mssa-loop-dependency -enable-simple-loop-unswitch -verify-memoryssa  < %s | FileCheck %s

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-ibm-linux"

@g_225 = external dso_local global i16, align 2
@g_967 = external dso_local global i8, align 2
@g_853 = external dso_local global i32***, align 8
@g_320 = external dso_local global { i8, i8, i8, i8, i8, i8, i8, i8 }, align 4

; Function Attrs: nounwind
; CHECK-LABEL: @main(
define dso_local void @main() #0 {
bb:
  call void @func_1()
  unreachable
}

; Function Attrs: nounwind
define dso_local void @func_1() #0 {
bb:
  call void @func_23()
  unreachable
}

; Function Attrs: nounwind
define dso_local void @func_23() #0 {
bb:
  %tmp = alloca i32****, align 8
  %tmp1 = alloca i32*****, align 8
  store i32**** @g_853, i32***** %tmp, align 8, !tbaa !1
  store i32***** %tmp, i32****** %tmp1, align 8, !tbaa !1
  br label %bb2

bb2:                                              ; preds = %bb21, %bb
  br label %bb3

bb3:                                              ; preds = %bb7, %bb2
  %tmp4 = load i8, i8* @g_967, align 2, !tbaa !5
  %tmp5 = sext i8 %tmp4 to i32
  %tmp6 = icmp sle i32 %tmp5, 5
  br i1 %tmp6, label %bb7, label %bb8

bb7:                                              ; preds = %bb3
  call void @safe_sub_func_uint64_t_u_u()
  br label %bb3

bb8:                                              ; preds = %bb3
  store i16 0, i16* @g_225, align 2, !tbaa !6
  br label %bb9

bb9:                                              ; preds = %bb25, %bb8
  %tmp10 = load i16, i16* @g_225, align 2, !tbaa !6
  %tmp11 = sext i16 %tmp10 to i32
  %tmp12 = icmp ne i32 %tmp11, 1
  br i1 %tmp12, label %bb13, label %bb28

bb13:                                             ; preds = %bb9
  %tmp14 = load i32*****, i32****** %tmp1, align 8, !tbaa !1
  %tmp15 = load i32****, i32***** %tmp14, align 8, !tbaa !1
  %tmp16 = load i32***, i32**** %tmp15, align 8, !tbaa !1
  %tmp17 = load i32**, i32*** %tmp16, align 8, !tbaa !1
  %tmp18 = load i32*, i32** %tmp17, align 8, !tbaa !1
  %tmp19 = load i32, i32* %tmp18, align 4, !tbaa !8
  %tmp20 = icmp ne i32 %tmp19, 0
  br i1 %tmp20, label %bb28, label %bb21

bb21:                                             ; preds = %bb13
  %tmp22 = load i32, i32* bitcast ({ i8, i8, i8, i8, i8, i8, i8, i8 }* @g_320 to i32*), align 4
  %tmp23 = ashr i32 %tmp22, 16
  %tmp24 = icmp ne i32 %tmp23, 0
  br i1 %tmp24, label %bb2, label %bb25

bb25:                                             ; preds = %bb21
  %tmp26 = load i16, i16* @g_225, align 2, !tbaa !6
  %tmp27 = add i16 %tmp26, 1
  store i16 %tmp27, i16* @g_225, align 2, !tbaa !6
  br label %bb9

bb28:                                             ; preds = %bb13, %bb9
  ret void
}

; Function Attrs: nounwind
define dso_local void @func_33() #0 {
bb:
  unreachable
}

; Function Attrs: nounwind
declare dso_local void @safe_sub_func_uint64_t_u_u() #0

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="z13" "target-features"="+transactional-execution,+vector" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 8.0.0 (http://llvm.org/git/clang.git a674a04e68bcf09f9a0423f3f589589596bc01a6) (http://llvm.org/git/llvm.git 1fe1ffe00e034128d1c5504254fdd4742f48bb9a)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!3, !3, i64 0}
!6 = !{!7, !7, i64 0}
!7 = !{!"short", !3, i64 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !3, i64 0}

