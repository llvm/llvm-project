; RUN: opt -mtriple=s390x-linux-gnu -mcpu=z13 -verify-memoryssa -passes='function(sroa),globalopt,function-attrs,function(simplifycfg,loop-mssa(licm),loop(simple-loop-unswitch))' %s -S | FileCheck %s
; REQUIRES: asserts

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-ibm-linux"

@0 = internal global i32 -9, align 4
@1 = internal global i64 9, align 8
@g_1042 = external dso_local global [5 x i16], align 2

; CHECK-LABEL: @main()
; Function Attrs: nounwind
define dso_local void @main() #0 {
  call void @func_1()
  unreachable
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #1

; Function Attrs: nounwind
define dso_local void @func_1() #0 {
  %1 = alloca ptr, align 8
  %2 = call signext i32 @func_2()
  %3 = icmp ne i32 %2, 0
  br i1 %3, label %4, label %9

; <label>:4:                                      ; preds = %0
  %5 = load i16, ptr @g_1042, align 2, !tbaa !1
  %6 = zext i16 %5 to i64
  %7 = load i64, ptr @1, align 8, !tbaa !5
  %8 = and i64 %7, %6
  store i64 %8, ptr @1, align 8, !tbaa !5
  call void @llvm.lifetime.end.p0(i64 4, ptr undef) #2
  unreachable

; <label>:9:                                      ; preds = %0
  store ptr @0, ptr %1, align 8, !tbaa !7
  br label %10

; <label>:10:                                     ; preds = %23, %9
  %11 = load i64, ptr @1, align 8, !tbaa !5
  %12 = icmp eq i64 %11, 65535
  br i1 %12, label %13, label %14

; <label>:13:                                     ; preds = %10
  store ptr null, ptr %1, align 8, !tbaa !7
  br label %14

; <label>:14:                                     ; preds = %13, %10
  %15 = load ptr, ptr %1, align 8, !tbaa !7
  %16 = load i32, ptr %15, align 4, !tbaa !9
  %17 = trunc i32 %16 to i16
  %18 = call signext i16 @safe_sub_func_int16_t_s_s(i16 signext %17)
  %19 = sext i16 %18 to i32
  %20 = icmp ne i32 %19, 0
  br i1 %20, label %23, label %21

; <label>:21:                                     ; preds = %14
  %22 = load volatile i8, ptr null, align 1, !tbaa !11
  br label %23

; <label>:23:                                     ; preds = %21, %14
  br label %10
}

; Function Attrs: nounwind
declare dso_local signext i32 @func_2() #0

; Function Attrs: nounwind
define dso_local void @safe_sub_func_uint8_t_u_u() #0 {
  ret void
}

; Function Attrs: nounwind
define dso_local void @safe_add_func_int64_t_s_s() #0 {
  ret void
}

; Function Attrs: nounwind
define dso_local void @safe_rshift_func_int16_t_s_u() #0 {
  ret void
}

; Function Attrs: nounwind
define dso_local void @safe_div_func_uint8_t_u_u() #0 {
  ret void
}

; Function Attrs: nounwind
define dso_local void @safe_mul_func_uint16_t_u_u() #0 {
  ret void
}

; Function Attrs: nounwind
define dso_local void @safe_mul_func_int16_t_s_s() #0 {
  ret void
}

; Function Attrs: nounwind
define dso_local void @safe_div_func_int32_t_s_s() #0 {
  ret void
}

; Function Attrs: nounwind
define dso_local signext i16 @safe_sub_func_int16_t_s_s(i16 signext) #0 {
  %2 = alloca i16, align 2
  store i16 %0, ptr %2, align 2, !tbaa !1
  %3 = load i16, ptr %2, align 2, !tbaa !1
  %4 = sext i16 %3 to i32
  %5 = sub nsw i32 %4, 0
  %6 = trunc i32 %5 to i16
  ret i16 %6
}

; Function Attrs: nounwind
define dso_local void @safe_add_func_uint16_t_u_u() #0 {
  ret void
}

; Function Attrs: nounwind
define dso_local void @safe_div_func_int8_t_s_s() #0 {
  ret void
}

; Function Attrs: nounwind
define dso_local void @safe_add_func_int16_t_s_s() #0 {
  ret void
}

; Function Attrs: nounwind
define dso_local void @safe_add_func_uint8_t_u_u() #0 {
  ret void
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="z13" "target-features"="+transactional-execution,+vector" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 8.0.0 (http://llvm.org/git/clang.git 7cda4756fc9713d98fd3513b8df172700f267bad) (http://llvm.org/git/llvm.git 199c0d32e96b646bd8cf6beeaf0f99f8a434b56a)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"short", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"long", !3, i64 0}
!7 = !{!8, !8, i64 0}
!8 = !{!"any pointer", !3, i64 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !3, i64 0}
!11 = !{!3, !3, i64 0}
