; RUN: opt < %s -S -mcpu=z13 -msan-kernel=1 -float-abi=soft -passes=msan 2>&1 | FileCheck %s

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-a:8:16-n32:64"
target triple = "s390x-unknown-linux-gnu"

declare i64 @foo(i64 %guard, ...) #0

attributes #0 = { "target-features"="+soft-float" "use-soft-float"="true" }

declare i32 @random_i32()
declare i64 @random_i64()
declare float @random_float()
declare double @random_double()

define i64 @bar() #1 {
  %arg2 = call i32 () @random_i32()
  %arg3 = call float () @random_float()
  %arg4 = call i32 () @random_i32()
  %arg5 = call double () @random_double()
  %arg6 = call i64 () @random_i64()
  %arg9 = call i32 () @random_i32()
  %arg11 = call float () @random_float()
  %arg12 = call i32 () @random_i32()
  %arg13 = call double () @random_double()
  %arg14 = call i64 () @random_i64()
  %1 = call i64 (i64, ...) @foo(i64 1, i32 zeroext %arg2, float %arg3,
                                i32 signext %arg4, double %arg5, i64 %arg6,
                                i64 7, double 8.0, i32 zeroext %arg9,
                                double 10.0, float %arg11, i32 signext %arg12,
                                double %arg13, i64 %arg14)
  ret i64 %1
}

attributes #1 = { sanitize_memory }

; In kernel the floating point values are passed in GPRs:
; - r2@16              == i64 1            - skipped, because it's fixed
; - r3@24              == i32 zext %arg2   - shadow is zero-extended
; - r4@(32 + 4)        == float %arg3      - right-justified, shadow is 32-bit
; - r5@40              == i32 sext %arg4   - shadow is sign-extended
; - r6@48              == double %arg5     - straightforward
; - overflow@160       == i64 %arg6        - straightforward
; - overflow@168       == 7                - filler
; - overflow@176       == 8.0              - filler
; - overflow@184       == i32 zext %arg9   - shadow is zero-extended
; - overflow@192       == 10.0             - filler
; - overflow@(200 + 4) == float %arg11     - right-justified, shadow is 32-bit
; - overflow@208       == i32 sext %arg12  - shadow is sign-extended
; - overflow@216       == double %arg13    - straightforward
; - overflow@224       == i64 %arg14       - straightforward
; Overflow arg area size is 72.

; CHECK-LABEL: @bar

; CHECK: [[B:%.*]] = ptrtoint [100 x i64]* %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 24
; CHECK: [[V:%.*]] = zext {{.*}}
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to i64*
; CHECK: store {{.*}} [[V]], {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint [100 x i64]* %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 36
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to i32*
; CHECK: store {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint [100 x i64]* %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 40
; CHECK: [[V:%.*]] = sext {{.*}}
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to i64*
; CHECK: store {{.*}} [[V]], {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint [100 x i64]* %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 48
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to i64*
; CHECK: store {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint [100 x i64]* %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 160
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to i64*
; CHECK: store {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint [100 x i64]* %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 168
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to i64*
; CHECK: store {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint [100 x i64]* %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 176
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to i64*
; CHECK: store {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint [100 x i64]* %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 184
; CHECK: [[V:%.*]] = zext {{.*}}
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to i64*
; CHECK: store {{.*}} [[V]], {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint [100 x i64]* %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 192
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to i64*
; CHECK: store {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint [100 x i64]* %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 204
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to i32*
; CHECK: store {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint [100 x i64]* %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 208
; CHECK: [[V:%.*]] = sext {{.*}}
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to i64*
; CHECK: store {{.*}} [[V]], {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint [100 x i64]* %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 216
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to i64*
; CHECK: store {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint [100 x i64]* %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 224
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to i64*
; CHECK: store {{.*}} [[M]]

; CHECK: store {{.*}} 72, {{.*}} %va_arg_overflow_size
