; RUN: opt < %s -passes=separate-const-offset-from-gep,slsr,gvn -S | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_35 | FileCheck %s --check-prefix=PTX
; RUN: opt < %s -passes="separate-const-offset-from-gep,slsr,gvn" -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-unknown"

; arr[i + 5]
; arr[i * 2 + 5]
; arr[i * 3 + 5]
; arr[i * 4 + 5]
;
;   => reassociate-geps
;
; *(&arr[i] + 5)
; *(&arr[i * 2] + 5)
; *(&arr[i * 3] + 5)
; *(&arr[i * 4] + 5)
;
;   => slsr
;
; p1 = &arr[i]
; *(p1 + 5)
; p2 = p1 + i
; *(p2 + 5)
; p3 = p2 + i
; *(p3 + 5)
; p4 = p3 + i
; *(p4 + 5)
define void @slsr_after_reassociate_geps(ptr %arr, i32 %i) {
; CHECK-LABEL: @slsr_after_reassociate_geps(
; PTX-LABEL: .visible .func slsr_after_reassociate_geps(
; PTX: ld.param.u64 [[arr:%rd[0-9]+]], [slsr_after_reassociate_geps_param_0];
; PTX: ld.param.u32 [[i:%r[0-9]+]], [slsr_after_reassociate_geps_param_1];
  %i2 = shl nsw i32 %i, 1
  %i3 = mul nsw i32 %i, 3
  %i4 = shl nsw i32 %i, 2

  %j1 = add nsw i32 %i, 5
  %p1 = getelementptr inbounds float, ptr %arr, i32 %j1
; CHECK: [[b1:%[0-9]+]] = getelementptr float, ptr %arr, i64 [[bump:%[0-9]+]]
; PTX: mul.wide.s32 [[i4:%rd[0-9]+]], [[i]], 4;
; PTX: add.s64 [[base1:%rd[0-9]+]], [[arr]], [[i4]];
  %v1 = load float, ptr %p1, align 4
; PTX: ld.f32 {{%f[0-9]+}}, [[[base1]]+20];
  call void @foo(float %v1)

  %j2 = add nsw i32 %i2, 5
  %p2 = getelementptr inbounds float, ptr %arr, i32 %j2
; CHECK: [[b2:%[0-9]+]] = getelementptr float, ptr [[b1]], i64 [[bump]]
; PTX: add.s64 [[base2:%rd[0-9]+]], [[base1]], [[i4]];
  %v2 = load float, ptr %p2, align 4
; PTX: ld.f32 {{%f[0-9]+}}, [[[base2]]+20];
  call void @foo(float %v2)

  %j3 = add nsw i32 %i3, 5
  %p3 = getelementptr inbounds float, ptr %arr, i32 %j3
; CHECK: [[b3:%[0-9]+]] = getelementptr float, ptr [[b2]], i64 [[bump]]
; PTX: add.s64 [[base3:%rd[0-9]+]], [[base2]], [[i4]];
  %v3 = load float, ptr %p3, align 4
; PTX: ld.f32 {{%f[0-9]+}}, [[[base3]]+20];
  call void @foo(float %v3)

  %j4 = add nsw i32 %i4, 5
  %p4 = getelementptr inbounds float, ptr %arr, i32 %j4
; CHECK: [[b4:%[0-9]+]] = getelementptr float, ptr [[b3]], i64 [[bump]]
; PTX: add.s64 [[base4:%rd[0-9]+]], [[base3]], [[i4]];
  %v4 = load float, ptr %p4, align 4
; PTX: ld.f32 {{%f[0-9]+}}, [[[base4]]+20];
  call void @foo(float %v4)

  ret void
}

declare void @foo(float)
