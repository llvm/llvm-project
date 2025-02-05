; RUN: llc < %s -mcpu=sm_35 | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; CUDA code
; __global__ void foo(int b, int s) {
;   #pragma unroll
;   for (int i = 0; i < 4; ++i) {
;     if (cond(i))
;       use((b + i) * s);
;   }
; }
define ptx_kernel void @foo(i32 %b, i32 %s) {
; CHECK-LABEL: .visible .entry foo(
entry:
; CHECK: ld.param.u32 [[s:%r[0-9]+]], [foo_param_1];
; CHECK: ld.param.u32 [[b:%r[0-9]+]], [foo_param_0];
  %call = tail call zeroext i1 @cond(i32 0)
  br i1 %call, label %if.then, label %for.inc

if.then:                                          ; preds = %entry
  %mul = mul nsw i32 %b, %s
; CHECK: mul.lo.s32 [[a0:%r[0-9]+]], [[b]], [[s]]
  tail call void @use(i32 %mul)
  br label %for.inc

for.inc:                                          ; preds = %entry, %if.then
  %call.1 = tail call zeroext i1 @cond(i32 1)
  br i1 %call.1, label %if.then.1, label %for.inc.1

if.then.1:                                        ; preds = %for.inc
  %add.1 = add nsw i32 %b, 1
  %mul.1 = mul nsw i32 %add.1, %s
; CHECK: add.s32 [[a1:%r[0-9]+]], [[a0]], [[s]]
  tail call void @use(i32 %mul.1)
  br label %for.inc.1

for.inc.1:                                        ; preds = %if.then.1, %for.inc
  %call.2 = tail call zeroext i1 @cond(i32 2)
  br i1 %call.2, label %if.then.2, label %for.inc.2

if.then.2:                                        ; preds = %for.inc.1
  %add.2 = add nsw i32 %b, 2
  %mul.2 = mul nsw i32 %add.2, %s
; CHECK: add.s32 [[a2:%r[0-9]+]], [[a1]], [[s]]
  tail call void @use(i32 %mul.2)
  br label %for.inc.2

for.inc.2:                                        ; preds = %if.then.2, %for.inc.1
  %call.3 = tail call zeroext i1 @cond(i32 3)
  br i1 %call.3, label %if.then.3, label %for.inc.3

if.then.3:                                        ; preds = %for.inc.2
  %add.3 = add nsw i32 %b, 3
  %mul.3 = mul nsw i32 %add.3, %s
; CHECK: add.s32 [[a3:%r[0-9]+]], [[a2]], [[s]]
  tail call void @use(i32 %mul.3)
  br label %for.inc.3

for.inc.3:                                        ; preds = %if.then.3, %for.inc.2
  ret void
}

declare zeroext i1 @cond(i32)

declare void @use(i32)
