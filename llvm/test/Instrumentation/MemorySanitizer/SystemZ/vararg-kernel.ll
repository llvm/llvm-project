; RUN: opt < %s -S -mcpu=z13 -msan-kernel=1 -float-abi=soft -passes=msan 2>&1 | FileCheck %s

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-a:8:16-n32:64"
target triple = "s390x-unknown-linux-gnu"

%struct.__va_list = type { i64, i64, ptr, ptr }
declare void @llvm.lifetime.start.p0(i64, ptr)
declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)
declare void @llvm.lifetime.end.p0(i64, ptr)

define i64 @foo(i64 %guard, ...) #1 {
  %vl = alloca %struct.__va_list
  call void @llvm.lifetime.start.p0(i64 32, ptr %vl)
  call void @llvm.va_start(ptr %vl)
  call void @llvm.va_end(ptr %vl)
  call void @llvm.lifetime.end.p0(i64 32, ptr %vl)
  ret i64 0
}

; CHECK-LABEL: define {{[^@]+}}@foo(

; Callers store variadic arguments' shadow and origins into va_arg_shadow and
; va_arg_origin. Their layout is: the register save area (160 bytes) followed
; by the overflow arg area. It does not depend on "packed-stack".
; Check that callees correctly backup shadow into a local variable.

; CHECK: [[TMP:%.*]] = alloca { ptr, ptr }
; CHECK: [[OverflowSize:%.*]] = load i64, ptr %va_arg_overflow_size
; CHECK: [[MetaSize:%.*]] = add i64 160, [[OverflowSize]]
; CHECK: [[ShadowBackup:%.*]] = alloca {{.*}} [[MetaSize]]
; CHECK: [[MetaCopySize:%.*]] = call i64 @llvm.umin.i64(i64 [[MetaSize]], i64 800)
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[ShadowBackup]], ptr align 8 %va_arg_shadow, i64 [[MetaCopySize]], i1 false)
; CHECK: [[OverflowBackup:%.*]] = alloca {{.*}} [[MetaSize]]
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[OverflowBackup]], ptr align 8 %va_arg_origin, i64 [[MetaCopySize]], i1 false)

; Check that va_start() correctly copies the shadow backup into the shadow of
; the va_list. Register save area and overflow arg area are copied separately.
; Only 56 bytes of the register save area is copied, because of
; "use-soft-float".

; CHECK: call void @llvm.va_start.p0(ptr %vl)
; CHECK: [[VlAddr:%.*]] = ptrtoint ptr %vl to i64
; CHECK: [[RegSaveAreaAddrAddr:%.*]] = add i64 [[VlAddr]], 24
; CHECK: [[RegSaveAreaAddr:%.*]] = inttoptr i64 [[RegSaveAreaAddrAddr]] to ptr
; CHECK: [[RegSaveArea:%.*]] = load ptr, ptr [[RegSaveAreaAddr]]
; CHECK: call void @__msan_metadata_ptr_for_store_1(ptr [[TMP]], ptr [[RegSaveArea]])
; CHECK: [[RegSaveAreaMeta:%.*]] = load { ptr, ptr }, ptr [[TMP]]
; CHECK: [[RegSaveAreaShadow:%.*]] = extractvalue { ptr, ptr } [[RegSaveAreaMeta]], 0
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RegSaveAreaShadow]], ptr align 8 [[ShadowBackup]], i64 56, i1 false)
; CHECK: [[VlAddr:%.*]] = ptrtoint ptr %vl to i64
; CHECK: [[OverflowAddrAddr:%.*]] = add i64 [[VlAddr]], 16
; CHECK: [[OverflowAddr:%.*]] = inttoptr i64 [[OverflowAddrAddr]] to ptr
; CHECK: [[Overflow:%.*]] = load ptr, ptr [[OverflowAddr]]
; CHECK: call void @__msan_metadata_ptr_for_store_1(ptr [[TMP]], ptr [[Overflow]])
; CHECK: [[OverflowMeta:%.*]] = load { ptr, ptr }, ptr [[TMP]]
; CHECK: [[OverflowShadow:%.*]] = extractvalue { ptr, ptr } [[OverflowMeta]], 0
; CHECK: [[OverflowShadowBackup:%.*]] = getelementptr i8, ptr [[ShadowBackup]], i32 160
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[OverflowShadow]], ptr align 8 [[OverflowShadowBackup]], i64 [[OverflowSize]], i1 false)

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

attributes #1 = { sanitize_memory "target-features"="+soft-float" "use-soft-float"="true" }

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

; CHECK: [[B:%.*]] = ptrtoint ptr %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 24
; CHECK: [[V:%.*]] = zext {{.*}}
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to ptr
; CHECK: store {{.*}} [[V]], {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint ptr %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 36
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to ptr
; CHECK: store {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint ptr %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 40
; CHECK: [[V:%.*]] = sext {{.*}}
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to ptr
; CHECK: store {{.*}} [[V]], {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint ptr %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 48
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to ptr
; CHECK: store {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint ptr %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 160
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to ptr
; CHECK: store {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint ptr %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 168
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to ptr
; CHECK: store {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint ptr %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 176
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to ptr
; CHECK: store {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint ptr %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 184
; CHECK: [[V:%.*]] = zext {{.*}}
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to ptr
; CHECK: store {{.*}} [[V]], {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint ptr %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 192
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to ptr
; CHECK: store {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint ptr %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 204
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to ptr
; CHECK: store {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint ptr %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 208
; CHECK: [[V:%.*]] = sext {{.*}}
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to ptr
; CHECK: store {{.*}} [[V]], {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint ptr %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 216
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to ptr
; CHECK: store {{.*}} [[M]]

; CHECK: [[B:%.*]] = ptrtoint ptr %va_arg_shadow to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 224
; CHECK: [[M:%_msarg_va_s.*]] = inttoptr i64 [[S]] to ptr
; CHECK: store {{.*}} [[M]]

; CHECK: store {{.*}} 72, {{.*}} %va_arg_overflow_size
