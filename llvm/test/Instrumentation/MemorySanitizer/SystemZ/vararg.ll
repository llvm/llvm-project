; RUN: opt < %s -S -mcpu=z13 -passes=msan 2>&1 | FileCheck %s

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-a:8:16-n32:64"
target triple = "s390x-unknown-linux-gnu"

%struct.__va_list = type { i64, i64, i8*, i8* }

define i64 @foo(i64 %guard, ...) {
  %vl = alloca %struct.__va_list, align 8
  %1 = bitcast %struct.__va_list* %vl to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* %1)
  call void @llvm.va_start(i8* %1)
  call void @llvm.va_end(i8* %1)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* %1)
  ret i64 0
}

; First check if the variadic shadow values are saved in stack with correct
; size (which is 160 - size of the register save area).

; CHECK-LABEL: @foo
; CHECK: [[A:%.*]] = load {{.*}} @__msan_va_arg_overflow_size_tls
; CHECK: [[B:%.*]] = add i64 160, [[A]]
; CHECK: alloca {{.*}} [[B]]

; We expect two memcpy operations: one for the register save area, and one for
; the overflow arg area.

; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 {{%.*}}, i8* align 8 {{%.*}}, i64 160, i1 false)
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 {{%.*}}, i8* align 8 {{%.*}}, i64 [[A]], i1 false)

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1
declare void @llvm.va_start(i8*) #2
declare void @llvm.va_end(i8*) #2
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

declare i32 @random_i32()
declare i64 @random_i64()
declare float @random_float()
declare double @random_double()

define i64 @bar() {
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

; Save the incoming shadow values from the varargs in the __msan_va_arg_tls
; array at the offsets equal to those defined by the ABI for the corresponding
; registers in the register save area, and for the corresponding overflow args
; in the overflow arg area:
; - r2@16              == i64 1            - skipped, because it's fixed
; - r3@24              == i32 zext %arg2   - shadow is zero-extended
; - f0@128             == float %arg3      - left-justified, shadow is 32-bit
; - r4@32              == i32 sext %arg4   - shadow is sign-extended
; - f2@136             == double %arg5     - straightforward
; - r5@40              == i64 %arg6        - straightforward
; - r6@48              == 7                - filler
; - f4@144             == 8.0              - filler
; - overflow@160       == i32 zext %arg9   - shadow is zero-extended
; - f6@152             == 10.0             - filler
; - overflow@(168 + 4) == float %arg11     - right-justified, shadow is 32-bit
; - overflow@176       == i32 sext %arg12  - shadow is sign-extended
; - overflow@184       == double %arg13    - straightforward
; - overflow@192       == i64 %arg14       - straightforward
; Overflow arg area size is 40.

; CHECK-LABEL: @bar

; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 24
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 128
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 32
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 136
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 40
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 48
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 144
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 160
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 152
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 172
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 176
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 184
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 192
; CHECK: store {{.*}} 40, {{.*}} @__msan_va_arg_overflow_size_tls

; Test that MSan doesn't generate code overflowing __msan_va_arg_tls when too many arguments are
; passed to a variadic function.

define dso_local i64 @many_args() {
entry:
  %ret = call i64 (i64, ...) @sum(i64 120,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1
  )
  ret i64 %ret
}

; If the size of __msan_va_arg_tls changes the second argument of `add` must also be changed.
; CHECK-LABEL: @many_args
; CHECK: i64 add (i64 ptrtoint ([100 x i64]* @__msan_va_arg_tls to i64), i64 792)
; CHECK-NOT: i64 add (i64 ptrtoint ([100 x i64]* @__msan_va_arg_tls to i64), i64 800)

declare i64 @sum(i64 %n, ...)

; Test offset calculation for vector arguments.
; Regardless of whether or not fixed args overflow, we should copy a shadow
; for a single vector vararg to offset 160.

declare void @vr_no_overflow(<4 x float> %v24, <4 x float> %v26,
                             <4 x float> %v28, <4 x float> %v30,
                             <4 x float> %v25, <4 x float> %v27,
                             <4 x float> %v29, ...)

declare <4 x float> @vr_value()

define void @vr_no_overflow_caller() {
  %1 = call <4 x float> () @vr_value()
  call void (<4 x float>, <4 x float>, <4 x float>,
             <4 x float>, <4 x float>, <4 x float>,
             <4 x float>, ...) @vr_no_overflow(
    <4 x float> %1, <4 x float> %1, <4 x float> %1, <4 x float> %1,
    <4 x float> %1, <4 x float> %1, <4 x float> %1, <4 x float> %1)
  ret void
}

; CHECK-LABEL: @vr_no_overflow_caller
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 160
; CHECK-NOT: store {{.*}} @__msan_va_arg_tls {{.*}}
; CHECK: store {{.*}} 16, {{.*}} @__msan_va_arg_overflow_size_tls

declare void @vr_overflow(<4 x float> %v24, <4 x float> %v26,
                          <4 x float> %v28, <4 x float> %v30,
                          <4 x float> %v25, <4 x float> %v27,
                          <4 x float> %v29, <4 x float> %v31,
                          <4 x float> %overflow, ...)

define void @vr_overflow_caller() {
  %1 = call <4 x float> @vr_value()
  call void (<4 x float>, <4 x float>, <4 x float>,
             <4 x float>, <4 x float>, <4 x float>,
             <4 x float>, <4 x float>, <4 x float>,
             ...) @vr_overflow(
    <4 x float> %1, <4 x float> %1, <4 x float> %1, <4 x float> %1,
    <4 x float> %1, <4 x float> %1, <4 x float> %1, <4 x float> %1,
    <4 x float> %1, <4 x float> %1)
  ret void
}

; CHECK-LABEL: @vr_overflow_caller
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 160
; CHECK-NOT: store {{.*}} @__msan_va_arg_tls {{.*}}
; CHECK: store {{.*}} 16, {{.*}} @__msan_va_arg_overflow_size_tls

; Test that i128 and fp128 are passed by reference.

declare i128 @random_i128()
declare fp128 @random_fp128()

define i64 @bar_128() {
  %iarg = call i128 @random_i128()
  %fparg = call fp128 @random_fp128()
  %1 = call i64 (i64, ...) @foo(i64 1, i128 %iarg, fp128 %fparg)
  ret i64 %1
}

; CHECK-LABEL: @bar_128
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 24
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 32
; CHECK: store {{.*}} 0, {{.*}} @__msan_va_arg_overflow_size_tls
