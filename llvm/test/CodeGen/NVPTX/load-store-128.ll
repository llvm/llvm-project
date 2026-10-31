; Codegen for plain and volatile 128-bit loads and stores (i128 and fp128) in
; every NVPTX address space. These are the non-atomic counterparts to
; atomic-load-store-128.ll: each access is split into a v2.b64 pair by
; SelectionDAG legalization. Volatile local accesses use plain loads/stores,
; as the .volatile modifier is only supported for generic, global, and shared
; memory.
;
; Checks are handwritten -- do not regenerate with update_llc_test_checks.py.

; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90  -mattr=+ptx83 | FileCheck %s
; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_100 -mattr=+ptx86 | FileCheck %s

target triple = "nvptx64-nvidia-cuda"


;; ---------------------------------------------------------------------------
;; i128
;; ---------------------------------------------------------------------------

; CHECK-LABEL: .visible .func {{.*}}ld_generic(
; CHECK: ld.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define i128 @ld_generic(ptr %p) {
  %v = load i128, ptr %p, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_generic_volatile(
; CHECK: ld.volatile.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define i128 @ld_generic_volatile(ptr %p) {
  %v = load volatile i128, ptr %p, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func st_generic(
; CHECK: st.v2.b64 [%rd{{[0-9]+}}], {{.*}};
define void @st_generic(ptr %p, i128 %v) {
  store i128 %v, ptr %p, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_generic_volatile(
; CHECK: st.volatile.v2.b64 [%rd{{[0-9]+}}], {{.*}};
define void @st_generic_volatile(ptr %p, i128 %v) {
  store volatile i128 %v, ptr %p, align 16
  ret void
}

; CHECK-LABEL: .visible .func {{.*}}ld_global(
; CHECK: ld.global.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define i128 @ld_global(ptr addrspace(1) %p) {
  %v = load i128, ptr addrspace(1) %p, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_global_volatile(
; CHECK: ld.volatile.global.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define i128 @ld_global_volatile(ptr addrspace(1) %p) {
  %v = load volatile i128, ptr addrspace(1) %p, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func st_global(
; CHECK: st.global.v2.b64 [%rd{{[0-9]+}}], {{.*}};
define void @st_global(ptr addrspace(1) %p, i128 %v) {
  store i128 %v, ptr addrspace(1) %p, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_global_volatile(
; CHECK: st.volatile.global.v2.b64 [%rd{{[0-9]+}}], {{.*}};
define void @st_global_volatile(ptr addrspace(1) %p, i128 %v) {
  store volatile i128 %v, ptr addrspace(1) %p, align 16
  ret void
}

; CHECK-LABEL: .visible .func {{.*}}ld_shared(
; CHECK: ld.shared.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define i128 @ld_shared(ptr addrspace(3) %p) {
  %v = load i128, ptr addrspace(3) %p, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_shared_volatile(
; CHECK: ld.volatile.shared.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define i128 @ld_shared_volatile(ptr addrspace(3) %p) {
  %v = load volatile i128, ptr addrspace(3) %p, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func st_shared(
; CHECK: st.shared.v2.b64 [%rd{{[0-9]+}}], {{.*}};
define void @st_shared(ptr addrspace(3) %p, i128 %v) {
  store i128 %v, ptr addrspace(3) %p, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_shared_volatile(
; CHECK: st.volatile.shared.v2.b64 [%rd{{[0-9]+}}], {{.*}};
define void @st_shared_volatile(ptr addrspace(3) %p, i128 %v) {
  store volatile i128 %v, ptr addrspace(3) %p, align 16
  ret void
}

; CHECK-LABEL: .visible .func {{.*}}ld_const(
; CHECK: ld.const.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define i128 @ld_const(ptr addrspace(4) %p) {
  %v = load i128, ptr addrspace(4) %p, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_const_volatile(
; CHECK: ld.const.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define i128 @ld_const_volatile(ptr addrspace(4) %p) {
  %v = load volatile i128, ptr addrspace(4) %p, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_local(
; CHECK: ld.local.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define i128 @ld_local(ptr addrspace(5) %p) {
  %v = load i128, ptr addrspace(5) %p, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_local_volatile(
; CHECK: ld.local.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define i128 @ld_local_volatile(ptr addrspace(5) %p) {
  %v = load volatile i128, ptr addrspace(5) %p, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func st_local(
; CHECK: st.local.v2.b64 [%rd{{[0-9]+}}], {{.*}};
define void @st_local(ptr addrspace(5) %p, i128 %v) {
  store i128 %v, ptr addrspace(5) %p, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_local_volatile(
; CHECK: st.local.v2.b64 [%rd{{[0-9]+}}], {{.*}};
define void @st_local_volatile(ptr addrspace(5) %p, i128 %v) {
  store volatile i128 %v, ptr addrspace(5) %p, align 16
  ret void
}

; CHECK-LABEL: .visible .func {{.*}}ld_shared_cluster(
; CHECK: ld.shared::cluster.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define i128 @ld_shared_cluster(ptr addrspace(7) %p) {
  %v = load i128, ptr addrspace(7) %p, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_shared_cluster_volatile(
; CHECK: ld.volatile.shared::cluster.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define i128 @ld_shared_cluster_volatile(ptr addrspace(7) %p) {
  %v = load volatile i128, ptr addrspace(7) %p, align 16
  ret i128 %v
}

; CHECK-LABEL: .visible .func st_shared_cluster(
; CHECK: st.shared::cluster.v2.b64 [%rd{{[0-9]+}}], {{.*}};
define void @st_shared_cluster(ptr addrspace(7) %p, i128 %v) {
  store i128 %v, ptr addrspace(7) %p, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_shared_cluster_volatile(
; CHECK: st.volatile.shared::cluster.v2.b64 [%rd{{[0-9]+}}], {{.*}};
define void @st_shared_cluster_volatile(ptr addrspace(7) %p, i128 %v) {
  store volatile i128 %v, ptr addrspace(7) %p, align 16
  ret void
}


;; ---------------------------------------------------------------------------
;; fp128
;; ---------------------------------------------------------------------------

; CHECK-LABEL: .visible .func {{.*}}ld_generic_f128(
; CHECK: ld.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define fp128 @ld_generic_f128(ptr %p) {
  %v = load fp128, ptr %p, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_generic_volatile_f128(
; CHECK: ld.volatile.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define fp128 @ld_generic_volatile_f128(ptr %p) {
  %v = load volatile fp128, ptr %p, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func st_generic_f128(
; CHECK: st.v2.b64 [%rd{{[0-9]+}}], {{.*}};
define void @st_generic_f128(ptr %p, fp128 %v) {
  store fp128 %v, ptr %p, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_generic_volatile_f128(
; CHECK: st.volatile.v2.b64 [%rd{{[0-9]+}}], {{.*}};
define void @st_generic_volatile_f128(ptr %p, fp128 %v) {
  store volatile fp128 %v, ptr %p, align 16
  ret void
}

; CHECK-LABEL: .visible .func {{.*}}ld_global_f128(
; CHECK: ld.global.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define fp128 @ld_global_f128(ptr addrspace(1) %p) {
  %v = load fp128, ptr addrspace(1) %p, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_global_volatile_f128(
; CHECK: ld.volatile.global.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define fp128 @ld_global_volatile_f128(ptr addrspace(1) %p) {
  %v = load volatile fp128, ptr addrspace(1) %p, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func st_global_f128(
; CHECK: st.global.v2.b64 [%rd{{[0-9]+}}], {{.*}};
define void @st_global_f128(ptr addrspace(1) %p, fp128 %v) {
  store fp128 %v, ptr addrspace(1) %p, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_global_volatile_f128(
; CHECK: st.volatile.global.v2.b64 [%rd{{[0-9]+}}], {{.*}};
define void @st_global_volatile_f128(ptr addrspace(1) %p, fp128 %v) {
  store volatile fp128 %v, ptr addrspace(1) %p, align 16
  ret void
}

; CHECK-LABEL: .visible .func {{.*}}ld_shared_f128(
; CHECK: ld.shared.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define fp128 @ld_shared_f128(ptr addrspace(3) %p) {
  %v = load fp128, ptr addrspace(3) %p, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_shared_volatile_f128(
; CHECK: ld.volatile.shared.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define fp128 @ld_shared_volatile_f128(ptr addrspace(3) %p) {
  %v = load volatile fp128, ptr addrspace(3) %p, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func st_shared_f128(
; CHECK: st.shared.v2.b64 [%rd{{[0-9]+}}], {{.*}};
define void @st_shared_f128(ptr addrspace(3) %p, fp128 %v) {
  store fp128 %v, ptr addrspace(3) %p, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_shared_volatile_f128(
; CHECK: st.volatile.shared.v2.b64 [%rd{{[0-9]+}}], {{.*}};
define void @st_shared_volatile_f128(ptr addrspace(3) %p, fp128 %v) {
  store volatile fp128 %v, ptr addrspace(3) %p, align 16
  ret void
}

; CHECK-LABEL: .visible .func {{.*}}ld_const_f128(
; CHECK: ld.const.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define fp128 @ld_const_f128(ptr addrspace(4) %p) {
  %v = load fp128, ptr addrspace(4) %p, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_const_volatile_f128(
; CHECK: ld.const.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define fp128 @ld_const_volatile_f128(ptr addrspace(4) %p) {
  %v = load volatile fp128, ptr addrspace(4) %p, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_local_f128(
; CHECK: ld.local.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define fp128 @ld_local_f128(ptr addrspace(5) %p) {
  %v = load fp128, ptr addrspace(5) %p, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_local_volatile_f128(
; CHECK: ld.local.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define fp128 @ld_local_volatile_f128(ptr addrspace(5) %p) {
  %v = load volatile fp128, ptr addrspace(5) %p, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func st_local_f128(
; CHECK: st.local.v2.b64 [%rd{{[0-9]+}}], {{.*}};
define void @st_local_f128(ptr addrspace(5) %p, fp128 %v) {
  store fp128 %v, ptr addrspace(5) %p, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_local_volatile_f128(
; CHECK: st.local.v2.b64 [%rd{{[0-9]+}}], {{.*}};
define void @st_local_volatile_f128(ptr addrspace(5) %p, fp128 %v) {
  store volatile fp128 %v, ptr addrspace(5) %p, align 16
  ret void
}

; CHECK-LABEL: .visible .func {{.*}}ld_shared_cluster_f128(
; CHECK: ld.shared::cluster.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define fp128 @ld_shared_cluster_f128(ptr addrspace(7) %p) {
  %v = load fp128, ptr addrspace(7) %p, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func {{.*}}ld_shared_cluster_volatile_f128(
; CHECK: ld.volatile.shared::cluster.v2.b64 {{.*}}[%rd{{[0-9]+}}];
define fp128 @ld_shared_cluster_volatile_f128(ptr addrspace(7) %p) {
  %v = load volatile fp128, ptr addrspace(7) %p, align 16
  ret fp128 %v
}

; CHECK-LABEL: .visible .func st_shared_cluster_f128(
; CHECK: st.shared::cluster.v2.b64 [%rd{{[0-9]+}}], {{.*}};
define void @st_shared_cluster_f128(ptr addrspace(7) %p, fp128 %v) {
  store fp128 %v, ptr addrspace(7) %p, align 16
  ret void
}

; CHECK-LABEL: .visible .func st_shared_cluster_volatile_f128(
; CHECK: st.volatile.shared::cluster.v2.b64 [%rd{{[0-9]+}}], {{.*}};
define void @st_shared_cluster_volatile_f128(ptr addrspace(7) %p, fp128 %v) {
  store volatile fp128 %v, ptr addrspace(7) %p, align 16
  ret void
}
