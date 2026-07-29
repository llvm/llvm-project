; RUN: opt < %s -passes='function(tsan),module(tsan-module)' -S | FileCheck %s
; REQUIRES: amdgpu-registered-target

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

; CHECK-NOT: @llvm.global_ctors = {{.*}}@tsan.module_ctor

; CHECK-LABEL: @entry_exit
; CHECK: call void @__tsan_kernel_entry()
; CHECK: call void @__tsan_func_entry(ptr %{{.*}})
; CHECK: call void @__tsan_read4(ptr %p)
; CHECK: call void @__tsan_func_exit()
define amdgpu_kernel void @entry_exit(ptr %p) sanitize_thread {
entry:
  %0 = load i32, ptr %p, align 4
  ret void
}

; CHECK-LABEL: @read_flat
define amdgpu_kernel void @read_flat(ptr %p) sanitize_thread {
entry:
; CHECK: call void @__tsan_read4(ptr %p)
  %0 = load i32, ptr %p, align 4
  ret void
}

; CHECK-LABEL: @write_flat
define amdgpu_kernel void @write_flat(ptr %p) sanitize_thread {
entry:
; CHECK: call void @__tsan_write4(ptr %p)
  store i32 1, ptr %p, align 4
  ret void
}

; CHECK-LABEL: @read_global
define amdgpu_kernel void @read_global(ptr addrspace(1) %p) sanitize_thread {
entry:
; CHECK: addrspacecast ptr addrspace(1) %p to ptr
; CHECK: call void @__tsan_read4(ptr %{{.*}})
  %0 = load i32, ptr addrspace(1) %p, align 4
  ret void
}

; CHECK-LABEL: @write_global
define amdgpu_kernel void @write_global(ptr addrspace(1) %p) sanitize_thread {
entry:
; CHECK: addrspacecast ptr addrspace(1) %p to ptr
; CHECK: call void @__tsan_write4(ptr %{{.*}})
  store i32 1, ptr addrspace(1) %p, align 4
  ret void
}

; CHECK-LABEL: @read_lds
define amdgpu_kernel void @read_lds(ptr addrspace(3) %p) sanitize_thread {
entry:
; CHECK: addrspacecast ptr addrspace(3) %p to ptr
; CHECK: call void @__tsan_read4(ptr %{{.*}})
  %0 = load i32, ptr addrspace(3) %p, align 4
  ret void
}

; CHECK-LABEL: @write_lds
define amdgpu_kernel void @write_lds(ptr addrspace(3) %p) sanitize_thread {
entry:
; CHECK: addrspacecast ptr addrspace(3) %p to ptr
; CHECK: call void @__tsan_write4(ptr %{{.*}})
  store i32 1, ptr addrspace(3) %p, align 4
  ret void
}

; CHECK-LABEL: @read_private
define amdgpu_kernel void @read_private(ptr addrspace(5) %p) sanitize_thread {
entry:
; CHECK-NOT: call void @__tsan
; CHECK: ret void
  %0 = load i32, ptr addrspace(5) %p, align 4
  ret void
}

; CHECK-LABEL: @read_constant
define amdgpu_kernel void @read_constant(ptr addrspace(4) %p) sanitize_thread {
entry:
; CHECK: addrspacecast ptr addrspace(4) %p to ptr
; CHECK: call void @__tsan_read4(ptr %{{.*}})
  %0 = load i32, ptr addrspace(4) %p, align 4
  ret void
}

; CHECK-LABEL: @atomic_load_system
define amdgpu_kernel void @atomic_load_system(ptr %p) sanitize_thread {
entry:
; CHECK: call i32 @__tsan_atomic32_load(ptr %p, i32 5, i32 0)
  %0 = load atomic i32, ptr %p syncscope("") seq_cst, align 4
  ret void
}

; CHECK-LABEL: @atomic_load_agent
define amdgpu_kernel void @atomic_load_agent(ptr %p) sanitize_thread {
entry:
; CHECK: call i32 @__tsan_atomic32_load(ptr %p, i32 2, i32 1)
  %0 = load atomic i32, ptr %p syncscope("agent") acquire, align 4
  ret void
}

; CHECK-LABEL: @atomic_load_workgroup
define amdgpu_kernel void @atomic_load_workgroup(ptr %p) sanitize_thread {
entry:
; CHECK: call i32 @__tsan_atomic32_load(ptr %p, i32 2, i32 2)
  %0 = load atomic i32, ptr %p syncscope("workgroup") acquire, align 4
  ret void
}

; CHECK-LABEL: @atomic_load_wavefront
define amdgpu_kernel void @atomic_load_wavefront(ptr %p) sanitize_thread {
entry:
; CHECK: call i32 @__tsan_atomic32_load(ptr %p, i32 0, i32 3)
  %0 = load atomic i32, ptr %p syncscope("wavefront") monotonic, align 4
  ret void
}

; CHECK-LABEL: @atomic_store_agent
define amdgpu_kernel void @atomic_store_agent(ptr %p) sanitize_thread {
entry:
; CHECK: call void @__tsan_atomic32_store(ptr %p, i32 42, i32 3, i32 1)
  store atomic i32 42, ptr %p syncscope("agent") release, align 4
  ret void
}

; CHECK-LABEL: @atomic_store_workgroup
define amdgpu_kernel void @atomic_store_workgroup(ptr %p) sanitize_thread {
entry:
; CHECK: call void @__tsan_atomic32_store(ptr %p, i32 7, i32 3, i32 2)
  store atomic i32 7, ptr %p syncscope("workgroup") release, align 4
  ret void
}

; CHECK-LABEL: @atomic_rmw_add_agent
define amdgpu_kernel void @atomic_rmw_add_agent(ptr %p) sanitize_thread {
entry:
; CHECK: call i32 @__tsan_atomic32_fetch_add(ptr %p, i32 1, i32 5, i32 1)
  %0 = atomicrmw add ptr %p, i32 1 syncscope("agent") seq_cst, align 4
  ret void
}

; CHECK-LABEL: @atomic_cas_workgroup
define amdgpu_kernel void @atomic_cas_workgroup(ptr %p) sanitize_thread {
entry:
; CHECK: call i32 @__tsan_atomic32_compare_exchange_val(ptr %p, i32 0, i32 1, i32 4, i32 2, i32 2)
  %0 = cmpxchg ptr %p, i32 0, i32 1 syncscope("workgroup") acq_rel acquire
  ret void
}

; CHECK-LABEL: @fence_agent
define amdgpu_kernel void @fence_agent() sanitize_thread {
entry:
; CHECK: call void @__tsan_atomic_thread_fence(i32 3, i32 1)
  fence syncscope("agent") release
  ret void
}

; CHECK-LABEL: @fence_workgroup
define amdgpu_kernel void @fence_workgroup() sanitize_thread {
entry:
; CHECK: call void @__tsan_atomic_thread_fence(i32 2, i32 2)
  fence syncscope("workgroup") acquire
  ret void
}

; CHECK-LABEL: @atomic_load_global_agent
define amdgpu_kernel void @atomic_load_global_agent(ptr addrspace(1) %p) sanitize_thread {
entry:
; CHECK: call i32 @__tsan_atomic32_load(ptr %{{.*}}, i32 2, i32 1)
  %0 = load atomic i32, ptr addrspace(1) %p syncscope("agent") acquire, align 4
  ret void
}

; CHECK-LABEL: @atomic_load_i64_agent
define amdgpu_kernel void @atomic_load_i64_agent(ptr %p) sanitize_thread {
entry:
; CHECK: call i64 @__tsan_atomic64_load(ptr %p, i32 2, i32 1)
  %0 = load atomic i64, ptr %p syncscope("agent") acquire, align 8
  ret void
}

; CHECK-LABEL: @read_i8_flat
define amdgpu_kernel void @read_i8_flat(ptr %p) sanitize_thread {
entry:
; CHECK: call void @__tsan_read1(ptr %p)
  %0 = load i8, ptr %p, align 1
  ret void
}

; CHECK-LABEL: @memcpy_flat
define amdgpu_kernel void @memcpy_flat(ptr %dst, ptr %src, i64 %n) sanitize_thread {
entry:
; CHECK: call ptr @__tsan_memcpy(ptr %dst, ptr %src, i64 %n)
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %n, i1 false)
  ret void
}

; CHECK-LABEL: @memcpy_private
define amdgpu_kernel void @memcpy_private(ptr addrspace(5) %dst, ptr addrspace(5) %src, i64 %n) sanitize_thread {
entry:
; CHECK: %[[DST:.*]] = addrspacecast ptr addrspace(5) %dst to ptr
; CHECK: %[[SRC:.*]] = addrspacecast ptr addrspace(5) %src to ptr
; CHECK: call ptr @__tsan_memcpy(ptr %[[DST]], ptr %[[SRC]], i64 %n)
  call void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) %dst, ptr addrspace(5) %src, i64 %n, i1 false)
  ret void
}

; CHECK-LABEL: @memmove_global
define amdgpu_kernel void @memmove_global(ptr addrspace(1) %dst, ptr addrspace(1) %src, i64 %n) sanitize_thread {
entry:
; CHECK: %[[DST:.*]] = addrspacecast ptr addrspace(1) %dst to ptr
; CHECK: %[[SRC:.*]] = addrspacecast ptr addrspace(1) %src to ptr
; CHECK: call ptr @__tsan_memmove(ptr %[[DST]], ptr %[[SRC]], i64 %n)
  call void @llvm.memmove.p1.p1.i64(ptr addrspace(1) %dst, ptr addrspace(1) %src, i64 %n, i1 false)
  ret void
}

; CHECK-LABEL: @memset_private
define amdgpu_kernel void @memset_private(ptr addrspace(5) %dst, i8 %c, i64 %n) sanitize_thread {
entry:
; CHECK: %[[DST:.*]] = addrspacecast ptr addrspace(5) %dst to ptr
; CHECK: call ptr @__tsan_memset(ptr %[[DST]], i32 %{{.*}}, i64 %n)
  call void @llvm.memset.p5.i64(ptr addrspace(5) %dst, i8 %c, i64 %n, i1 false)
  ret void
}
