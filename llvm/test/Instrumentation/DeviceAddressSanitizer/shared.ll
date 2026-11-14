; RUN: opt < %s -passes=dasan -S | FileCheck %s
; RUN: opt < %s -passes=dasan -dasan-instrument-shared=0 -S | FileCheck %s --check-prefix=OFF

target datalayout = "e-p:64:64:64-p3:32:32-i8:8:8-i16:16:16-i32:32:32-i64:64:64-n8:16:32:64-S128-A5-G1"
target triple = "amdgcn-amd-amdhsa"

@vis = addrspace(3) global [4 x i32] poison
; CHECK: @vis = addrspace(3) global [4 x i32] poison

@ext = weak addrspace(3) global [32 x i32] poison
; CHECK: @ext = weak addrspace(3) global [32 x i32] poison

@dyn = internal addrspace(3) global [0 x i32] poison
; CHECK: @dyn = internal addrspace(3) global [0 x i32] poison

@kept = internal addrspace(3) global [4 x i32] poison
; CHECK: @kept = internal addrspace(3) global [4 x i32] poison

@llvm.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(3) @kept to ptr)], section "llvm.metadata"

@a = internal addrspace(3) global [64 x i32] poison
@b = internal addrspace(3) global [16 x i32] poison
@slot = internal addrspace(3) global ptr addrspace(3) poison
; CHECK: @a = internal addrspace(3) global [64 x i32] poison
; CHECK: @b = internal addrspace(3) global [16 x i32] poison
; CHECK-NOT: @__dasan_shared =
; CHECK-NOT: @__dasan_lds_bound
; OFF: @a = internal addrspace(3) global [64 x i32] poison
; OFF: @b = internal addrspace(3) global [16 x i32] poison
; OFF-NOT: @__dasan_lds_bound

define void @in_bounds() sanitize_device_address {
; CHECK-LABEL: define void @in_bounds()
; CHECK-NEXT: store i32 1, ptr addrspace(3) @a
; CHECK-NEXT: ret void
  store i32 1, ptr addrspace(3) @a
  ret void
}

define void @dynamic(i32 %i) sanitize_device_address {
; CHECK-LABEL: define void @dynamic(
; CHECK: [[PKT:%.*]] = call {{.*}}@llvm.amdgcn.dispatch.ptr()
; CHECK: [[GEP:%.*]] = getelementptr inbounds i8, ptr addrspace(4) [[PKT]], i64 28
; CHECK: [[LIMIT:%.*]] = load i32, ptr addrspace(4) [[GEP]], align 4, !invariant.load
; CHECK: icmp uge i32 {{.*}}, [[LIMIT]]
; CHECK-NOT: ballot
; CHECK: call void @__dasan_report_shared_store(
; CHECK: store i32 1, ptr addrspace(3) %{{.*}}
  %p = getelementptr [0 x i32], ptr addrspace(3) @dyn, i32 0, i32 %i
  store i32 1, ptr addrspace(3) %p
  ret void
}

; The pass needs the dispatch packet for unsized LDS. Drop the attribute rather
; than skipping the check.
define void @dynamic_no_dispatch(i32 %i) sanitize_device_address "amdgpu-no-dispatch-ptr" {
; CHECK-LABEL: define void @dynamic_no_dispatch(
; CHECK-NOT: amdgpu-no-dispatch-ptr
; CHECK: call {{.*}}@llvm.amdgcn.dispatch.ptr
; CHECK: call void @__dasan_report_shared_store(
; CHECK: store i32 1, ptr addrspace(3) %{{.*}}
  %p = getelementptr [0 x i32], ptr addrspace(3) @dyn, i32 0, i32 %i
  store i32 1, ptr addrspace(3) %p
  ret void
}

define void @past_end_no_dispatch() sanitize_device_address "amdgpu-no-dispatch-ptr" {
; CHECK-LABEL: define void @past_end_no_dispatch(
; CHECK-NOT: call {{.*}}@llvm.amdgcn.dispatch.ptr
; CHECK: call void @__dasan_report_shared_store(ptr {{.*}}, i64 4, i64 {{.*}}, i64 256)
  %p = getelementptr [64 x i32], ptr addrspace(3) @a, i32 0, i32 64
  store i32 1, ptr addrspace(3) %p
  ret void
}

define void @flat_dyn(i32 %i) sanitize_device_address {
; CHECK-LABEL: define void @flat_dyn(
; CHECK: call {{.*}}@llvm.amdgcn.dispatch.ptr
; CHECK-NOT: lshr i64 {{.*}}, 45
; CHECK: call void @__dasan_report_shared_store(
  %p = addrspacecast ptr addrspace(3) @dyn to ptr
  %q = getelementptr i32, ptr %p, i32 %i
  store i32 1, ptr %q
  ret void
}

define void @past_end() sanitize_device_address {
; CHECK-LABEL: define void @past_end()
; CHECK: call void @__dasan_report_shared_store(ptr {{.*}}, i64 4, i64 {{.*}}, i64 256)
  %p = getelementptr [64 x i32], ptr addrspace(3) @a, i32 0, i32 64
  store i32 1, ptr addrspace(3) %p
  ret void
}

; PHI: getUnderlyingObject is the PHI, so the check is the group-segment limit.
define void @phi(i1 %c, i32 %i) sanitize_device_address {
; CHECK-LABEL: define void @phi(
; CHECK: call {{.*}}@llvm.amdgcn.dispatch.ptr
; CHECK: store i32 1, ptr addrspace(3) %p
  br i1 %c, label %l, label %r
l:
  %pa = getelementptr [64 x i32], ptr addrspace(3) @a, i32 0, i32 %i
  br label %j
r:
  %pb = getelementptr [16 x i32], ptr addrspace(3) @b, i32 0, i32 %i
  br label %j
j:
  %p = phi ptr addrspace(3) [ %pa, %l ], [ %pb, %r ]
  store i32 1, ptr addrspace(3) %p
  ret void
}

; Load from private: no underlying global.
define void @spill(i32 %i) sanitize_device_address {
; CHECK-LABEL: define void @spill(
; CHECK: call {{.*}}@llvm.amdgcn.dispatch.ptr
; CHECK: store i32 1, ptr addrspace(3) %q
  %s = alloca ptr addrspace(3), addrspace(5)
  %p = getelementptr [64 x i32], ptr addrspace(3) @a, i32 0, i32 %i
  store ptr addrspace(3) %p, ptr addrspace(5) %s
  %q = load ptr addrspace(3), ptr addrspace(5) %s
  store i32 1, ptr addrspace(3) %q
  ret void
}

define void @callee(ptr addrspace(3) %p) sanitize_device_address {
; CHECK-LABEL: define void @callee(
; CHECK: call {{.*}}@llvm.amdgcn.dispatch.ptr
; CHECK: store i32 1, ptr addrspace(3) %p
  store i32 1, ptr addrspace(3) %p
  ret void
}

define void @caller() sanitize_device_address {
; CHECK-LABEL: define void @caller()
; CHECK: call void @callee(ptr addrspace(3) @a)
  call void @callee(ptr addrspace(3) @a)
  ret void
}

define void @weak_past() sanitize_device_address {
; CHECK-LABEL: define void @weak_past()
; CHECK: call void @__dasan_report_shared_store(ptr {{.*}}, i64 4, i64 {{.*}}, i64 128)
  %p = getelementptr [32 x i32], ptr addrspace(3) @ext, i32 0, i32 32
  store i32 1, ptr addrspace(3) %p
  ret void
}

; Decay does not hide the global from getUnderlyingObject.
define void @flat_of_local(i32 %i) sanitize_device_address {
; CHECK-LABEL: define void @flat_of_local(
; CHECK: icmp uge i32 {{.*}}, 256
; CHECK-NOT: call i1 @llvm.amdgcn.is.shared
; CHECK-NOT: lshr i64 {{.*}}, 45
; CHECK: call void @__dasan_report_shared_store(
  %p = addrspacecast ptr addrspace(3) @a to ptr
  %q = getelementptr i32, ptr %p, i32 %i
  store i32 1, ptr %q
  ret void
}

define void @past_end_optnone() sanitize_device_address optnone noinline {
; CHECK-LABEL: define void @past_end_optnone()
; CHECK: call void @__dasan_report_shared_store(ptr {{.*}}, i64 4, i64 {{.*}}, i64 256)
; CHECK: store i32 1, ptr addrspace(3) %p
  %p = getelementptr [64 x i32], ptr addrspace(3) @a, i32 0, i32 64
  store i32 1, ptr addrspace(3) %p
  ret void
}

define void @spill_optnone(i32 %i) sanitize_device_address optnone noinline {
; CHECK-LABEL: define void @spill_optnone(
; CHECK: call {{.*}}@llvm.amdgcn.dispatch.ptr
; CHECK: store i32 1, ptr addrspace(3) %q
  %s = alloca ptr addrspace(3), addrspace(5)
  %p = getelementptr [64 x i32], ptr addrspace(3) @a, i32 0, i32 %i
  store ptr addrspace(3) %p, ptr addrspace(5) %s
  %q = load ptr addrspace(3), ptr addrspace(5) %s
  store i32 1, ptr addrspace(3) %q
  ret void
}

define ptr addrspace(3) @gives() sanitize_device_address {
; CHECK-LABEL: define ptr addrspace(3) @gives()
; CHECK: ret ptr addrspace(3) @a
  ret ptr addrspace(3) @a
}

declare void @sink(ptr addrspace(3))

define void @escape_unsanitized() sanitize_device_address {
; CHECK-LABEL: define void @escape_unsanitized()
; CHECK: call void @sink(ptr addrspace(3) @a)
  call void @sink(ptr addrspace(3) @a)
  ret void
}

define void @callee_flat(ptr %p, i32 %i) sanitize_device_address {
; CHECK-LABEL: define void @callee_flat(
; CHECK-NOT: call {{.*}}@llvm.amdgcn.dispatch.ptr
; CHECK-NOT: call i1 @llvm.amdgcn.is.shared
; CHECK: lshr i64 {{.*}}, 45
  %q = getelementptr i32, ptr %p, i32 %i
  store i32 1, ptr %q
  ret void
}

define void @caller_flat(i32 %i) sanitize_device_address {
; CHECK-LABEL: define void @caller_flat(
; CHECK: call void @callee_flat(ptr %p, i32 %i)
  %p = addrspacecast ptr addrspace(3) @a to ptr
  call void @callee_flat(ptr %p, i32 %i)
  ret void
}

define void @callee_off(ptr %p) sanitize_device_address {
; CHECK-LABEL: define void @callee_off(
; CHECK-NOT: call {{.*}}@llvm.amdgcn.dispatch.ptr
; CHECK: lshr i64 {{.*}}, 45
  store i32 1, ptr %p
  ret void
}

define void @caller_decayed_gep(i32 %i) sanitize_device_address {
; CHECK-LABEL: define void @caller_decayed_gep(
; CHECK: call void @callee_off(ptr %q)
  %p = addrspacecast ptr addrspace(3) @a to ptr
  %q = getelementptr i32, ptr %p, i32 %i
  call void @callee_off(ptr %q)
  ret void
}

define amdgpu_kernel void @kernel_heap(ptr %out, i32 %i) sanitize_device_address {
; CHECK-LABEL: define amdgpu_kernel void @kernel_heap(
; CHECK-NOT: call {{.*}}@llvm.amdgcn.dispatch.ptr
; CHECK-NOT: call i1 @llvm.amdgcn.is.shared
; CHECK: %p = getelementptr i32, ptr %out, i32 %i
; CHECK: lshr i64 {{.*}}, 45
  %p = getelementptr i32, ptr %out, i32 %i
  store i32 1, ptr %p
  ret void
}

define i1 @cmp_spill(i32 %i) sanitize_device_address {
; CHECK-LABEL: define i1 @cmp_spill(
; CHECK: icmp eq ptr addrspace(3) %q, @a
  %s = alloca ptr addrspace(3), addrspace(5)
  %p = getelementptr [64 x i32], ptr addrspace(3) @a, i32 0, i32 %i
  store ptr addrspace(3) %p, ptr addrspace(5) %s
  %q = load ptr addrspace(3), ptr addrspace(5) %s
  %c = icmp eq ptr addrspace(3) %q, @a
  ret i1 %c
}

define void @spill_to_lds(i32 %i) sanitize_device_address {
; CHECK-LABEL: define void @spill_to_lds(
; CHECK: call {{.*}}@llvm.amdgcn.dispatch.ptr
; CHECK: store i32 1, ptr addrspace(3) %q
  %p = getelementptr [64 x i32], ptr addrspace(3) @a, i32 0, i32 %i
  store ptr addrspace(3) %p, ptr addrspace(3) @slot
  %q = load ptr addrspace(3), ptr addrspace(3) @slot
  store i32 1, ptr addrspace(3) %q
  ret void
}

; Escaped GEP of @a, then a load: the store is not a GEP of @a.
define void @escaped_oob() sanitize_device_address {
; CHECK-LABEL: define void @escaped_oob(
; CHECK: call {{.*}}@llvm.amdgcn.dispatch.ptr
; CHECK: store i32 1, ptr addrspace(3) %q
  %s = alloca ptr addrspace(3), addrspace(5)
  %p = getelementptr [64 x i32], ptr addrspace(3) @a, i32 0, i32 64
  store ptr addrspace(3) %p, ptr addrspace(5) %s
  %q = load ptr addrspace(3), ptr addrspace(5) %s
  store i32 1, ptr addrspace(3) %q
  ret void
}

; OFF-LABEL: define void @dynamic(
; OFF-NEXT: %p = getelementptr
; OFF-NEXT: store i32 1, ptr addrspace(3) %p
; OFF-NEXT: ret void
