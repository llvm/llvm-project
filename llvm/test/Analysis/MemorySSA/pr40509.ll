; REQUIRES: asserts
; RUN: opt -mtriple=systemz-unknown -mcpu=z13 -O3 -disable-output %s

; During transform to LCSSA, an access becomes obfuscated to:
; (2 = phi (phi(val), val)), which BasicAA fails to analyze.
; It's currently hard coded in BasicAA to return MayAlias for nested phis.
; This leads MemorySSA to finding a new (false) clobber for a previously
; optimized access. With verifyClobber included in verifyMemorySSA, such a
; transformation will cause MemorySSA verification to fail.
; If the verifyClobber is re-enabled, this test will crash.

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-ibm-linux"

%0 = type <{ i64, i8, i64, i16 }>

@g_54 = external dso_local global i16, align 2
@g_101 = external dso_local global <{ i64, i8, i64, i8, i8 }>, align 2

declare dso_local void @safe_lshift_func_int16_t_s_s()
declare dso_local i8 @safe_div_func_int8_t_s_s()

define dso_local void @func_47(%0* %arg) {
bb:
  %tmp = alloca i32, align 4
  br label %bb1

bb1:                                              ; preds = %bb12, %bb
  %tmp2 = getelementptr inbounds %0, %0* %arg, i32 0, i32 3
  store i16 undef, i16* %tmp2, align 1
  %tmp3 = call signext i8 @safe_div_func_int8_t_s_s()
  %tmp7 = icmp ne i8 %tmp3, 0
  br i1 %tmp7, label %bb8, label %bb10

bb8:                                              ; preds = %bb1
  %tmp9 = icmp eq i32 0, 0
  br i1 %tmp9, label %bb12, label %bb13

bb10:                                             ; preds = %bb10, %bb1
  call void @safe_lshift_func_int16_t_s_s()
  %tmp11 = getelementptr inbounds %0, %0* %arg, i32 0, i32 3
  store i16 0, i16* %tmp11, align 1
  store i8 0, i8* getelementptr inbounds (%0, %0* bitcast (<{ i64, i8, i64, i8, i8 }>* @g_101 to %0*), i32 0, i32 1), align 2
  br label %bb10

bb12:                                             ; preds = %bb8
  store i16 0, i16* @g_54, align 2
  br label %bb1

bb13:                                             ; preds = %bb8
  ret void
}


