; XFAIL: *
; RUN: opt -march=z13 -S -O3 -enable-simple-loop-unswitch -enable-mssa-loop-dependency -enable-loop-simplifycfg-term-folding 2>&1 < %s | FileCheck %s
; CHECK-LABEL: @main(

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"

@g_45 = external dso_local global i8, align 2
@g_182 = external dso_local global i32, align 4
@g_277 = external dso_local global i32, align 4
@g_1135 = external dso_local global i16, align 2
@g_2998 = external dso_local global i32, align 4

; Function Attrs: nounwind
define dso_local signext i32 @main(i32 signext %arg, i8** %arg1) #0 {
bb:
  %tmp = call signext i32 @0()
  unreachable
}

; Function Attrs: nounwind
define internal signext i32 @0() #0 {
bb:
  %tmp = call signext i32 @1(i32 zeroext 0, i64 undef)
  ret i32 undef
}

; Function Attrs: nounwind
define internal signext i32 @1(i32 zeroext %arg, i64 %arg1) #0 {
bb:
  %tmp = alloca i32, align 4
  %tmp2 = alloca i32, align 4
  store i32 %arg, i32* %tmp, align 4, !tbaa !1
  br label %bb3

bb3:                                              ; preds = %bb42, %bb
  store i32 48, i32* %tmp2, align 4
  %tmp4 = load i32, i32* %tmp2, align 4
  %tmp5 = icmp eq i32 %tmp4, 48
  br i1 %tmp5, label %bb6, label %bb42

bb6:                                              ; preds = %bb9, %bb3
  %tmp7 = load i32, i32* @g_277, align 4, !tbaa !1
  %tmp8 = icmp ule i32 %tmp7, 0
  br i1 %tmp8, label %bb9, label %bb16

bb9:                                              ; preds = %bb15, %bb6
  %tmp10 = icmp sle i32 0, 5
  %tmp11 = load i32, i32* %tmp, align 4, !tbaa !1
  br i1 %tmp10, label %bb12, label %bb6

bb12:                                             ; preds = %bb9
  %tmp13 = icmp ne i32 %tmp11, 0
  br i1 %tmp13, label %bb15, label %bb14

bb14:                                             ; preds = %bb12
  store i16 0, i16* @g_1135, align 2, !tbaa !5
  br label %bb15

bb15:                                             ; preds = %bb14, %bb12
  br label %bb9

bb16:                                             ; preds = %bb20, %bb6
  %tmp17 = load i32, i32* %tmp, align 4, !tbaa !1
  %tmp18 = icmp ule i32 %tmp17, 0
  br i1 %tmp18, label %bb19, label %bb22

bb19:                                             ; preds = %bb19, %bb16
  br i1 undef, label %bb19, label %bb20

bb20:                                             ; preds = %bb19
  %tmp21 = add i32 0, 1
  store i32 %tmp21, i32* %tmp, align 4, !tbaa !1
  br label %bb16

bb22:                                             ; preds = %bb40, %bb16
  %tmp23 = load i32, i32* @g_277, align 4, !tbaa !1
  %tmp24 = icmp ule i32 %tmp23, 5
  br i1 %tmp24, label %bb25, label %bb42

bb25:                                             ; preds = %bb22
  store i32 0, i32* @g_182, align 4, !tbaa !1
  br label %bb26

bb26:                                             ; preds = %bb29, %bb25
  %tmp27 = load i32, i32* @g_182, align 4, !tbaa !1
  %tmp28 = icmp ule i32 %tmp27, 0
  br i1 %tmp28, label %bb29, label %bb31

bb29:                                             ; preds = %bb26
  %tmp30 = load i32*, i32** undef, align 8, !tbaa !7
  br i1 undef, label %bb26, label %bb40

bb31:                                             ; preds = %bb35, %bb26
  %tmp32 = load i32, i32* @g_2998, align 4, !tbaa !1
  %tmp33 = icmp sle i32 %tmp32, 5
  br i1 %tmp33, label %bb34, label %bb39

bb34:                                             ; preds = %bb34, %bb31
  br i1 undef, label %bb34, label %bb35

bb35:                                             ; preds = %bb35, %bb34
  %tmp36 = load i8, i8* @g_45, align 2, !tbaa !9
  %tmp37 = zext i8 %tmp36 to i32
  %tmp38 = icmp sle i32 %tmp37, 5
  br i1 %tmp38, label %bb35, label %bb31

bb39:                                             ; preds = %bb31
  store i32 0, i32* %tmp2, align 4
  br label %bb40

bb40:                                             ; preds = %bb39, %bb29
  %tmp41 = icmp eq i32 0, 0
  br i1 %tmp41, label %bb22, label %bb42

bb42:                                             ; preds = %bb40, %bb22, %bb3
  %tmp43 = load i32, i32* %tmp2, align 4
  %tmp44 = icmp eq i32 %tmp43, 0
  br i1 %tmp44, label %bb3, label %bb45

bb45:                                             ; preds = %bb42
  ret i32 undef
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="z13" "target-features"="+transactional-execution,+vector" "unsafe-fp-math"="false" "use-soft-float"="false" }

!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"short", !3, i64 0}
!7 = !{!8, !8, i64 0}
!8 = !{!"any pointer", !3, i64 0}
!9 = !{!3, !3, i64 0}
