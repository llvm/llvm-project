; XFAIL: *
; RUN: opt  -S -march=z13 -O3 -enable-simple-loop-unswitch -enable-mssa-loop-dependency -enable-loop-simplifycfg-term-folding 2>&1 < %s | FileCheck %s
; CHECK-LABEL: @safe_sub_func_uint64_t_u_u(

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"

@g_57 = external dso_local global i16, align 2

; Function Attrs: nounwind
define dso_local void @main() #0 {
bb:
  call void @func_1()
  unreachable
}

; Function Attrs: nounwind
define dso_local void @func_1() #0 {
bb:
  call void @func_25()
  unreachable
}

; Function Attrs: nounwind
declare dso_local void @safe_lshift_func_int16_t_s_u() #0

; Function Attrs: nounwind
declare dso_local void @safe_mul_func_int8_t_s_s() #0

; Function Attrs: nounwind
define dso_local void @func_25() #0 {
bb:
  call void @func_50()
  unreachable
}

; Function Attrs: nounwind
define dso_local void @func_50() #0 {
bb:
  call void @func_76(i8 signext undef)
  unreachable
}

; Function Attrs: nounwind
define dso_local void @safe_lshift_func_uint8_t_u_u() #0 {
bb:
  ret void
}

; Function Attrs: nounwind
declare dso_local i64 @safe_sub_func_uint64_t_u_u() #0

; Function Attrs: nounwind
declare dso_local void @safe_mod_func_uint64_t_u_u() #0

; Function Attrs: nounwind
define dso_local void @func_76(i8 signext %arg) #0 {
bb:
  %tmp = alloca i8, align 1
  %tmp1 = alloca i32, align 4
  %tmp2 = alloca i32, align 4
  store i8 %arg, i8* %tmp, align 1, !tbaa !1
  br label %bb3

bb3:                                              ; preds = %bb29, %bb
  br label %bb4

bb4:                                              ; preds = %bb28, %bb3
  %tmp5 = load i16, i16* @g_57, align 2, !tbaa !4
  %tmp6 = zext i16 %tmp5 to i32
  %tmp7 = icmp sgt i32 %tmp6, 56
  br i1 %tmp7, label %bb8, label %bb29

bb8:                                              ; preds = %bb4
  %tmp9 = call i64 @safe_sub_func_uint64_t_u_u()
  %tmp10 = icmp ne i64 %tmp9, 0
  br i1 %tmp10, label %bb11, label %bb25

bb11:                                             ; preds = %bb8
  store i32 26, i32* %tmp1, align 4, !tbaa !6
  br label %bb12

bb12:                                             ; preds = %bb15, %bb11
  %tmp13 = load i32, i32* %tmp1, align 4, !tbaa !6
  %tmp14 = icmp ne i32 %tmp13, 12
  br i1 %tmp14, label %bb15, label %bb18

bb15:                                             ; preds = %bb12
  store i8 -23, i8* %tmp, align 1, !tbaa !1
  call void @safe_mod_func_uint64_t_u_u()
  %tmp16 = load i32, i32* %tmp1, align 4, !tbaa !6
  %tmp17 = add nsw i32 %tmp16, -1
  store i32 %tmp17, i32* %tmp1, align 4, !tbaa !6
  br label %bb12

bb18:                                             ; preds = %bb12
  %tmp19 = load i8, i8* %tmp, align 1, !tbaa !1
  %tmp20 = icmp ne i8 %tmp19, 0
  br i1 %tmp20, label %bb21, label %bb22

bb21:                                             ; preds = %bb18
  store i32 7, i32* %tmp2, align 4
  br label %bb22

bb22:                                             ; preds = %bb21, %bb18
  %tmp23 = load i32, i32* %tmp2, align 4
  %tmp24 = icmp eq i32 %tmp23, 0
  br i1 %tmp24, label %bb26, label %bb27

bb25:                                             ; preds = %bb8
  call void @safe_lshift_func_int16_t_s_u()
  store i32 0, i32* %tmp2, align 4
  switch i32 undef, label %bb30 [
    i32 0, label %bb26
    i32 7, label %bb28
  ]

bb26:                                             ; preds = %bb25, %bb22
  call void @safe_mul_func_int8_t_s_s()
  br label %bb27

bb27:                                             ; preds = %bb26, %bb22
  switch i32 undef, label %bb30 [
    i32 0, label %bb28
    i32 7, label %bb28
  ]

bb28:                                             ; preds = %bb27, %bb27, %bb25
  br label %bb4

bb29:                                             ; preds = %bb4
  br i1 undef, label %bb3, label %bb30

bb30:                                             ; preds = %bb29, %bb27, %bb25
  ret void
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="z13" "target-features"="+transactional-execution,+vector" "unsafe-fp-math"="false" "use-soft-float"="false" }

!1 = !{!2, !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"short", !2, i64 0}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !2, i64 0}
