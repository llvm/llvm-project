; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_50 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_50 | %ptxas-verify %}

; The expansion of small memcpy/memmove/memset can raise the alignment of
; under-aligned stack objects. Verify this also works on NVPTX, which lowers
; allocas to the local address space via addrspacecast instructions that hide
; the frame index, both when the pointer is materialized in the same basic
; block as the memory operation and when it is defined in a different one.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare void @use(ptr)

define void @memcpy_alloca() {
; CHECK-LABEL: memcpy_alloca(
; CHECK: .local .align 4 .b8 __local_depot{{[0-9]+}}[16];
; CHECK: st.local.b8
; CHECK: st.local.b16
; CHECK: st.local.b32
; CHECK-NOT: st.local.b8
; CHECK: ret;
  %a = alloca [7 x i8], align 1
  %b = alloca [7 x i8], align 1
  call void @use(ptr %a)
  call void @use(ptr %b)
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a, ptr align 1 %b, i64 7, i1 false)
  call void @use(ptr %a)
  ret void
}

define void @memcpy_alloca_cross_bb(i1 %c) {
; CHECK-LABEL: memcpy_alloca_cross_bb(
; CHECK: .local .align 4 .b8 __local_depot{{[0-9]+}}[16];
; CHECK: st.local.b8
; CHECK: st.local.b16
; CHECK: st.local.b32
; CHECK-NOT: st.local.b8
; CHECK: ret;
entry:
  %a = alloca [7 x i8], align 1
  %b = alloca [7 x i8], align 1
  call void @use(ptr %a)
  call void @use(ptr %b)
  br i1 %c, label %copy, label %exit

copy:
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a, ptr align 1 %b, i64 7, i1 false)
  call void @use(ptr %a)
  br label %exit

exit:
  ret void
}

; The source object's alignment was raised when expanding the first memcpy;
; the loads of the second copy can only learn about it through the frame index.
define void @memcpy_alloca_raised_src(i1 %c, ptr %p) {
; CHECK-LABEL: memcpy_alloca_raised_src(
; CHECK: .local .align 4 .b8 __local_depot{{[0-9]+}}[16];
; CHECK: ld.local.b8
; CHECK: ld.local.b16
; CHECK: ld.local.b32
; CHECK-NOT: ld.local.b8
; CHECK: ret;
entry:
  %a = alloca [7 x i8], align 1
  %b = alloca [7 x i8], align 1
  call void @use(ptr %a)
  call void @use(ptr %b)
  br i1 %c, label %copy1, label %exit

copy1:
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b, ptr align 1 %p, i64 7, i1 false)
  br label %copy2

copy2:
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a, ptr align 1 %b, i64 7, i1 false)
  call void @use(ptr %a)
  br label %exit

exit:
  ret void
}

define void @memmove_alloca() {
; CHECK-LABEL: memmove_alloca(
; CHECK: .local .align 4 .b8 __local_depot{{[0-9]+}}[16];
; CHECK: st.local.b8
; CHECK: st.local.b16
; CHECK: st.local.b32
; CHECK-NOT: st.local.b8
; CHECK: ret;
  %a = alloca [7 x i8], align 1
  %b = alloca [7 x i8], align 1
  call void @use(ptr %a)
  call void @use(ptr %b)
  call void @llvm.memmove.p0.p0.i64(ptr align 1 %a, ptr align 1 %b, i64 7, i1 false)
  call void @use(ptr %a)
  ret void
}

define void @memset_alloca() {
; CHECK-LABEL: memset_alloca(
; CHECK: .local .align 4 .b8 __local_depot{{[0-9]+}}[8];
; CHECK: st.local.b8
; CHECK: st.local.b16
; CHECK: st.local.b32
; CHECK-NOT: st.local.b8
; CHECK: ret;
  %a = alloca [7 x i8], align 1
  call void @use(ptr %a)
  call void @llvm.memset.p0.i64(ptr align 1 %a, i8 0, i64 7, i1 false)
  call void @use(ptr %a)
  ret void
}

; InferPtrAlign also looks through the addrspacecast, letting DAGCombiner
; refine the alignment of under-aligned plain loads and stores.
define i32 @underaligned_load_store(i32 %v) {
; CHECK-LABEL: underaligned_load_store(
; CHECK: ld.local.b32
; CHECK: st.local.b32
; CHECK-NOT: .local.b8
; CHECK: ret;
  %a = alloca i32, align 4
  call void @use(ptr %a)
  %l = load i32, ptr %a, align 1
  store i32 %v, ptr %a, align 1
  call void @use(ptr %a)
  ret i32 %l
}
