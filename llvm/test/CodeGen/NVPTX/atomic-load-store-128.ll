; Checks are handwritten -- do not regenerate with update_llc_test_checks.py.

; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_70  -mattr=+ptx84 | FileCheck %s --check-prefixes=CHECK,NOCLUSTER
; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_75  -mattr=+ptx84 | FileCheck %s --check-prefixes=CHECK,NOCLUSTER
; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_80  -mattr=+ptx84 | FileCheck %s --check-prefixes=CHECK,NOCLUSTER
; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90  -mattr=+ptx84 | FileCheck %s --check-prefixes=CHECK,CLUSTER
; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_100 -mattr=+ptx86 | FileCheck %s --check-prefixes=CHECK,CLUSTER
; RUN: %if ptxas-sm_90 && ptxas-isa-8.4 %{ llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90 -mattr=+ptx84 | %ptxas-verify -arch=sm_90 %}

; RUN: not llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_60 -mattr=+ptx83 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_70 -mattr=+ptx82 2>&1 | FileCheck %s --check-prefix=ERROR

; ERROR: error: unsupported atomic load: target supports atomics up to 8 bytes, but this atomic accesses 16 bytes
; ERROR: error: unsupported atomic store: target supports atomics up to 8 bytes, but this atomic accesses 16 bytes

target triple = "nvptx64-nvidia-cuda"

; CHECK-LABEL: .visible .func {{.*}}ld_generic_relaxed(
; CHECK: ld.relaxed.sys.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define i128 @ld_generic_relaxed(ptr %p) {
  %v = load atomic i128, ptr %p monotonic, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_generic_acquire(
; CHECK: ld.acquire.sys.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define i128 @ld_generic_acquire(ptr %p) {
  %v = load atomic i128, ptr %p acquire, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_global_relaxed(
; CHECK: ld.relaxed.sys.global.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define i128 @ld_global_relaxed(ptr addrspace(1) %p) {
  %v = load atomic i128, ptr addrspace(1) %p monotonic, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_global_acquire(
; CHECK: ld.acquire.sys.global.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define i128 @ld_global_acquire(ptr addrspace(1) %p) {
  %v = load atomic i128, ptr addrspace(1) %p acquire, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_shared_relaxed(
; CHECK: ld.relaxed.sys.shared.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define i128 @ld_shared_relaxed(ptr addrspace(3) %p) {
  %v = load atomic i128, ptr addrspace(3) %p monotonic, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_shared_acquire(
; CHECK: ld.acquire.sys.shared.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define i128 @ld_shared_acquire(ptr addrspace(3) %p) {
  %v = load atomic i128, ptr addrspace(3) %p acquire, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_shared_cluster_relaxed(
; CHECK: ld.relaxed.sys.shared::cluster.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define i128 @ld_shared_cluster_relaxed(ptr addrspace(7) %p) {
  %v = load atomic i128, ptr addrspace(7) %p monotonic, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_shared_cluster_acquire(
; CHECK: ld.acquire.sys.shared::cluster.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define i128 @ld_shared_cluster_acquire(ptr addrspace(7) %p) {
  %v = load atomic i128, ptr addrspace(7) %p acquire, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func st_generic_relaxed(
; CHECK: st.relaxed.sys.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_generic_relaxed(ptr %p, i128 %v) {
  store atomic i128 %v, ptr %p monotonic, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_generic_release(
; CHECK: st.release.sys.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_generic_release(ptr %p, i128 %v) {
  store atomic i128 %v, ptr %p release, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_global_relaxed(
; CHECK: st.relaxed.sys.global.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_global_relaxed(ptr addrspace(1) %p, i128 %v) {
  store atomic i128 %v, ptr addrspace(1) %p monotonic, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_global_release(
; CHECK: st.release.sys.global.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_global_release(ptr addrspace(1) %p, i128 %v) {
  store atomic i128 %v, ptr addrspace(1) %p release, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_shared_relaxed(
; CHECK: st.relaxed.sys.shared.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_shared_relaxed(ptr addrspace(3) %p, i128 %v) {
  store atomic i128 %v, ptr addrspace(3) %p monotonic, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_shared_release(
; CHECK: st.release.sys.shared.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_shared_release(ptr addrspace(3) %p, i128 %v) {
  store atomic i128 %v, ptr addrspace(3) %p release, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_shared_cluster_relaxed(
; CHECK: st.relaxed.sys.shared::cluster.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_shared_cluster_relaxed(ptr addrspace(7) %p, i128 %v) {
  store atomic i128 %v, ptr addrspace(7) %p monotonic, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_shared_cluster_release(
; CHECK: st.release.sys.shared::cluster.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_shared_cluster_release(ptr addrspace(7) %p, i128 %v) {
  store atomic i128 %v, ptr addrspace(7) %p release, align 16
  ret void
}

; CHECK-LABEL: .visible .func {{.*}}ld_generic_relaxed_f128(
; CHECK: ld.relaxed.sys.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define fp128 @ld_generic_relaxed_f128(ptr %p) {
  %v = load atomic fp128, ptr %p monotonic, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_generic_acquire_f128(
; CHECK: ld.acquire.sys.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define fp128 @ld_generic_acquire_f128(ptr %p) {
  %v = load atomic fp128, ptr %p acquire, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_global_relaxed_f128(
; CHECK: ld.relaxed.sys.global.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define fp128 @ld_global_relaxed_f128(ptr addrspace(1) %p) {
  %v = load atomic fp128, ptr addrspace(1) %p monotonic, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_global_acquire_f128(
; CHECK: ld.acquire.sys.global.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define fp128 @ld_global_acquire_f128(ptr addrspace(1) %p) {
  %v = load atomic fp128, ptr addrspace(1) %p acquire, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_shared_relaxed_f128(
; CHECK: ld.relaxed.sys.shared.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define fp128 @ld_shared_relaxed_f128(ptr addrspace(3) %p) {
  %v = load atomic fp128, ptr addrspace(3) %p monotonic, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_shared_acquire_f128(
; CHECK: ld.acquire.sys.shared.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define fp128 @ld_shared_acquire_f128(ptr addrspace(3) %p) {
  %v = load atomic fp128, ptr addrspace(3) %p acquire, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_shared_cluster_relaxed_f128(
; CHECK: ld.relaxed.sys.shared::cluster.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define fp128 @ld_shared_cluster_relaxed_f128(ptr addrspace(7) %p) {
  %v = load atomic fp128, ptr addrspace(7) %p monotonic, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_shared_cluster_acquire_f128(
; CHECK: ld.acquire.sys.shared::cluster.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define fp128 @ld_shared_cluster_acquire_f128(ptr addrspace(7) %p) {
  %v = load atomic fp128, ptr addrspace(7) %p acquire, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func st_generic_relaxed_f128(
; CHECK: st.relaxed.sys.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_generic_relaxed_f128(ptr %p, fp128 %v) {
  store atomic fp128 %v, ptr %p monotonic, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_generic_release_f128(
; CHECK: st.release.sys.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_generic_release_f128(ptr %p, fp128 %v) {
  store atomic fp128 %v, ptr %p release, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_global_relaxed_f128(
; CHECK: st.relaxed.sys.global.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_global_relaxed_f128(ptr addrspace(1) %p, fp128 %v) {
  store atomic fp128 %v, ptr addrspace(1) %p monotonic, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_global_release_f128(
; CHECK: st.release.sys.global.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_global_release_f128(ptr addrspace(1) %p, fp128 %v) {
  store atomic fp128 %v, ptr addrspace(1) %p release, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_shared_relaxed_f128(
; CHECK: st.relaxed.sys.shared.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_shared_relaxed_f128(ptr addrspace(3) %p, fp128 %v) {
  store atomic fp128 %v, ptr addrspace(3) %p monotonic, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_shared_release_f128(
; CHECK: st.release.sys.shared.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_shared_release_f128(ptr addrspace(3) %p, fp128 %v) {
  store atomic fp128 %v, ptr addrspace(3) %p release, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_shared_cluster_relaxed_f128(
; CHECK: st.relaxed.sys.shared::cluster.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_shared_cluster_relaxed_f128(ptr addrspace(7) %p, fp128 %v) {
  store atomic fp128 %v, ptr addrspace(7) %p monotonic, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_shared_cluster_release_f128(
; CHECK: st.release.sys.shared::cluster.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_shared_cluster_release_f128(ptr addrspace(7) %p, fp128 %v) {
  store atomic fp128 %v, ptr addrspace(7) %p release, align 16
  ret void
}

; CHECK-LABEL: .visible .func {{.*}}ld_generic_seq_cst(
; CHECK: fence.sc.sys;
; CHECK: ld.acquire.sys.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define i128 @ld_generic_seq_cst(ptr %p) {
  %v = load atomic i128, ptr %p seq_cst, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}st_generic_seq_cst(
; CHECK: fence.sc.sys;
; CHECK: st.relaxed.sys.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_generic_seq_cst(ptr %p, i128 %v) {
  store atomic i128 %v, ptr %p seq_cst, align 16
  ret void
}

; CHECK-LABEL: .visible .func {{.*}}ld_global_seq_cst(
; CHECK: fence.sc.sys;
; CHECK: ld.acquire.sys.global.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define i128 @ld_global_seq_cst(ptr addrspace(1) %p) {
  %v = load atomic i128, ptr addrspace(1) %p seq_cst, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}st_global_seq_cst(
; CHECK: fence.sc.sys;
; CHECK: st.relaxed.sys.global.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_global_seq_cst(ptr addrspace(1) %p, i128 %v) {
  store atomic i128 %v, ptr addrspace(1) %p seq_cst, align 16
  ret void
}

; CHECK-LABEL: .visible .func {{.*}}ld_shared_seq_cst(
; CHECK: fence.sc.sys;
; CHECK: ld.acquire.sys.shared.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define i128 @ld_shared_seq_cst(ptr addrspace(3) %p) {
  %v = load atomic i128, ptr addrspace(3) %p seq_cst, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}st_shared_seq_cst(
; CHECK: fence.sc.sys;
; CHECK: st.relaxed.sys.shared.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_shared_seq_cst(ptr addrspace(3) %p, i128 %v) {
  store atomic i128 %v, ptr addrspace(3) %p seq_cst, align 16
  ret void
}

; CHECK-LABEL: .visible .func {{.*}}ld_shared_cluster_seq_cst(
; CHECK: fence.sc.sys;
; CHECK: ld.acquire.sys.shared::cluster.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define i128 @ld_shared_cluster_seq_cst(ptr addrspace(7) %p) {
  %v = load atomic i128, ptr addrspace(7) %p seq_cst, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}st_shared_cluster_seq_cst(
; CHECK: fence.sc.sys;
; CHECK: st.relaxed.sys.shared::cluster.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_shared_cluster_seq_cst(ptr addrspace(7) %p, i128 %v) {
  store atomic i128 %v, ptr addrspace(7) %p seq_cst, align 16
  ret void
}

; CHECK-LABEL: .visible .func {{.*}}ld_generic_seq_cst_f128(
; CHECK: fence.sc.sys;
; CHECK: ld.acquire.sys.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define fp128 @ld_generic_seq_cst_f128(ptr %p) {
  %v = load atomic fp128, ptr %p seq_cst, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func {{.*}}st_generic_seq_cst_f128(
; CHECK: fence.sc.sys;
; CHECK: st.relaxed.sys.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_generic_seq_cst_f128(ptr %p, fp128 %v) {
  store atomic fp128 %v, ptr %p seq_cst, align 16
  ret void
}

; CHECK-LABEL: .visible .func {{.*}}ld_global_seq_cst_f128(
; CHECK: fence.sc.sys;
; CHECK: ld.acquire.sys.global.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define fp128 @ld_global_seq_cst_f128(ptr addrspace(1) %p) {
  %v = load atomic fp128, ptr addrspace(1) %p seq_cst, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func {{.*}}st_global_seq_cst_f128(
; CHECK: fence.sc.sys;
; CHECK: st.relaxed.sys.global.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_global_seq_cst_f128(ptr addrspace(1) %p, fp128 %v) {
  store atomic fp128 %v, ptr addrspace(1) %p seq_cst, align 16
  ret void
}

; CHECK-LABEL: .visible .func {{.*}}ld_shared_seq_cst_f128(
; CHECK: fence.sc.sys;
; CHECK: ld.acquire.sys.shared.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define fp128 @ld_shared_seq_cst_f128(ptr addrspace(3) %p) {
  %v = load atomic fp128, ptr addrspace(3) %p seq_cst, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func {{.*}}st_shared_seq_cst_f128(
; CHECK: fence.sc.sys;
; CHECK: st.relaxed.sys.shared.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_shared_seq_cst_f128(ptr addrspace(3) %p, fp128 %v) {
  store atomic fp128 %v, ptr addrspace(3) %p seq_cst, align 16
  ret void
}

; CHECK-LABEL: .visible .func {{.*}}ld_shared_cluster_seq_cst_f128(
; CHECK: fence.sc.sys;
; CHECK: ld.acquire.sys.shared::cluster.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define fp128 @ld_shared_cluster_seq_cst_f128(ptr addrspace(7) %p) {
  %v = load atomic fp128, ptr addrspace(7) %p seq_cst, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func {{.*}}st_shared_cluster_seq_cst_f128(
; CHECK: fence.sc.sys;
; CHECK: st.relaxed.sys.shared::cluster.b128 [%rd{{[0-9]+}}], %rq{{[0-9]+}};
define void @st_shared_cluster_seq_cst_f128(ptr addrspace(7) %p, fp128 %v) {
  store atomic fp128 %v, ptr addrspace(7) %p seq_cst, align 16
  ret void
}

; CHECK-LABEL: .visible .func {{.*}}ld_generic_seq_cst_block(
; CHECK: fence.sc.cta;
; CHECK: ld.acquire.cta.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define i128 @ld_generic_seq_cst_block(ptr %p) {
  %v = load atomic i128, ptr %p syncscope("block") seq_cst, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_generic_seq_cst_cluster(
; CLUSTER:   fence.sc.cluster;
; CLUSTER:   ld.acquire.cluster.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
; NOCLUSTER: fence.sc.cta;
; NOCLUSTER: ld.acquire.cta.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define i128 @ld_generic_seq_cst_cluster(ptr %p) {
  %v = load atomic i128, ptr %p syncscope("cluster") seq_cst, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_generic_seq_cst_device(
; CHECK: fence.sc.gpu;
; CHECK: ld.acquire.gpu.b128 %rq{{[0-9]+}}, [%rd{{[0-9]+}}];
define i128 @ld_generic_seq_cst_device(ptr %p) {
  %v = load atomic i128, ptr %p syncscope("device") seq_cst, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_generic_volatile_relaxed(
; CHECK: ld.volatile.b128
define i128 @ld_generic_volatile_relaxed(ptr %p) {
  %v = load atomic volatile i128, ptr %p monotonic, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_global_volatile_relaxed(
; CHECK: ld.mmio.relaxed.sys.global.b128
define i128 @ld_global_volatile_relaxed(ptr addrspace(1) %p) {
  %v = load atomic volatile i128, ptr addrspace(1) %p monotonic, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_shared_volatile_relaxed(
; CHECK: ld.volatile.shared.b128
define i128 @ld_shared_volatile_relaxed(ptr addrspace(3) %p) {
  %v = load atomic volatile i128, ptr addrspace(3) %p monotonic, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_global_volatile_acquire(
; CHECK: ld.acquire.sys.global.b128
define i128 @ld_global_volatile_acquire(ptr addrspace(1) %p) {
  %v = load atomic volatile i128, ptr addrspace(1) %p acquire, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_global_volatile_seq_cst(
; CHECK: fence.sc.sys;
; CHECK: ld.acquire.sys.global.b128
define i128 @ld_global_volatile_seq_cst(ptr addrspace(1) %p) {
  %v = load atomic volatile i128, ptr addrspace(1) %p seq_cst, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}st_generic_volatile_relaxed(
; CHECK: st.volatile.b128
define void @st_generic_volatile_relaxed(ptr %p, i128 %v) {
  store atomic volatile i128 %v, ptr %p monotonic, align 16
  ret void
}

; CHECK-LABEL: .visible .func {{.*}}st_global_volatile_relaxed(
; CHECK: st.mmio.relaxed.sys.global.b128
define void @st_global_volatile_relaxed(ptr addrspace(1) %p, i128 %v) {
  store atomic volatile i128 %v, ptr addrspace(1) %p monotonic, align 16
  ret void
}

; CHECK-LABEL: .visible .func {{.*}}st_shared_volatile_relaxed(
; CHECK: st.volatile.shared.b128
define void @st_shared_volatile_relaxed(ptr addrspace(3) %p, i128 %v) {
  store atomic volatile i128 %v, ptr addrspace(3) %p monotonic, align 16
  ret void
}

; CHECK-LABEL: .visible .func {{.*}}st_global_volatile_release(
; CHECK: st.release.sys.global.b128
define void @st_global_volatile_release(ptr addrspace(1) %p, i128 %v) {
  store atomic volatile i128 %v, ptr addrspace(1) %p release, align 16
  ret void
}

; CHECK-LABEL: .visible .func {{.*}}st_global_volatile_seq_cst(
; CHECK: fence.sc.sys;
; CHECK: st.mmio.relaxed.sys.global.b128
define void @st_global_volatile_seq_cst(ptr addrspace(1) %p, i128 %v) {
  store atomic volatile i128 %v, ptr addrspace(1) %p seq_cst, align 16
  ret void
}
