; RUN: llc < %s -march=nvptx64 -mcpu=sm_90 -mattr=+ptx78 | FileCheck %s
; RUN: %if ptxas-12.2 %{ llc < %s -march=nvptx64 -mcpu=sm_90 -mattr=+ptx78 | %ptxas-verify -arch=sm_90 %}

; TODO: fix "atomic load volatile acquire": generates "ld.acquire.sys;"
;       but should generate "ld.mmio.relaxed.sys; fence.acq_rel.sys;"
; TODO: fix "atomic store volatile release": generates "st.release.sys;"
;       but should generate "fence.acq_rel.sys; st.mmio.relaxed.sys;"

; TODO: fix "atomic load volatile seq_cst": generates "fence.sc.sys; ld.acquire.sys;"
;       but should generate "fence.sc.sys; ld.relaxed.mmio.sys; fence.acq_rel.sys;"
; TODO: fix "atomic store volatile seq_cst": generates "fence.sc.sys; st.release.sys;"
;       but should generate "fence.sc.sys; st.relaxed.mmio.sys;"

; TODO: add i1, <8 x i8>, and <6 x i8> vector tests.

; TODO: add test for vectors that exceed 128-bit length
; Per https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#vectors
; vectors cannot exceed 128-bit in length, i.e., .v4.u64 is not allowed.

; TODO: generate PTX that preserves Concurrent Forward Progress
;       for atomic operations to local statespace
;       by generating atomic or volatile operations.

; TODO: design exposure for atomic operations on vector types.

; TODO: implement and test thread scope.

; TODO: add weak,atomic,volatile,atomic volatile tests
;       for .const and .param statespaces.

; TODO: optimize .shared.sys into .shared.cta or .shared.cluster .

;; generic statespace

; CHECK-LABEL: generic_unordered_cluster
define void @generic_unordered_cluster(ptr %a, ptr %b, ptr %c, ptr %d, ptr %e) local_unnamed_addr {
  ; CHECK: ld.relaxed.cluster.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr %a syncscope("cluster") unordered, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.relaxed.cluster.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr %a syncscope("cluster") unordered, align 1

  ; CHECK: ld.relaxed.cluster.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr %b syncscope("cluster") unordered, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.relaxed.cluster.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr %b syncscope("cluster") unordered, align 2

  ; CHECK: ld.relaxed.cluster.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr %c syncscope("cluster") unordered, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.relaxed.cluster.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr %c syncscope("cluster") unordered, align 4

  ; CHECK: ld.relaxed.cluster.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr %d syncscope("cluster") unordered, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.relaxed.cluster.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr %d syncscope("cluster") unordered, align 8

  ; CHECK: ld.relaxed.cluster.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr %e syncscope("cluster") unordered, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.relaxed.cluster.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr %e syncscope("cluster") unordered, align 4

  ; CHECK: ld.relaxed.cluster.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr %e syncscope("cluster") unordered, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.relaxed.cluster.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr %e syncscope("cluster") unordered, align 8

  ret void
}

; CHECK-LABEL: generic_unordered_volatile_cluster
define void @generic_unordered_volatile_cluster(ptr %a, ptr %b, ptr %c, ptr %d, ptr %e) local_unnamed_addr {
  ; CHECK: ld.volatile.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr %a syncscope("cluster") unordered, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.volatile.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr %a syncscope("cluster") unordered, align 1

  ; CHECK: ld.volatile.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr %b syncscope("cluster") unordered, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.volatile.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr %b syncscope("cluster") unordered, align 2

  ; CHECK: ld.volatile.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr %c syncscope("cluster") unordered, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.volatile.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr %c syncscope("cluster") unordered, align 4

  ; CHECK: ld.volatile.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr %d syncscope("cluster") unordered, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.volatile.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr %d syncscope("cluster") unordered, align 8

  ; CHECK: ld.volatile.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr %e syncscope("cluster") unordered, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.volatile.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr %e syncscope("cluster") unordered, align 4

  ; CHECK: ld.volatile.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr %e syncscope("cluster") unordered, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.volatile.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr %e syncscope("cluster") unordered, align 8

  ret void
}

; CHECK-LABEL: generic_monotonic_cluster
define void @generic_monotonic_cluster(ptr %a, ptr %b, ptr %c, ptr %d, ptr %e) local_unnamed_addr {
  ; CHECK: ld.relaxed.cluster.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr %a syncscope("cluster") monotonic, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.relaxed.cluster.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr %a syncscope("cluster") monotonic, align 1

  ; CHECK: ld.relaxed.cluster.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr %b syncscope("cluster") monotonic, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.relaxed.cluster.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr %b syncscope("cluster") monotonic, align 2

  ; CHECK: ld.relaxed.cluster.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr %c syncscope("cluster") monotonic, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.relaxed.cluster.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr %c syncscope("cluster") monotonic, align 4

  ; CHECK: ld.relaxed.cluster.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr %d syncscope("cluster") monotonic, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.relaxed.cluster.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr %d syncscope("cluster") monotonic, align 8

  ; CHECK: ld.relaxed.cluster.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr %e syncscope("cluster") monotonic, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.relaxed.cluster.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr %e syncscope("cluster") monotonic, align 4

  ; CHECK: ld.relaxed.cluster.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr %e syncscope("cluster") monotonic, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.relaxed.cluster.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr %e syncscope("cluster") monotonic, align 8

  ret void
}

; CHECK-LABEL: generic_monotonic_volatile_cluster
define void @generic_monotonic_volatile_cluster(ptr %a, ptr %b, ptr %c, ptr %d, ptr %e) local_unnamed_addr {
  ; CHECK: ld.volatile.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr %a syncscope("cluster") monotonic, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.volatile.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr %a syncscope("cluster") monotonic, align 1

  ; CHECK: ld.volatile.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr %b syncscope("cluster") monotonic, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.volatile.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr %b syncscope("cluster") monotonic, align 2

  ; CHECK: ld.volatile.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr %c syncscope("cluster") monotonic, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.volatile.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr %c syncscope("cluster") monotonic, align 4

  ; CHECK: ld.volatile.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr %d syncscope("cluster") monotonic, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.volatile.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr %d syncscope("cluster") monotonic, align 8

  ; CHECK: ld.volatile.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr %e syncscope("cluster") monotonic, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.volatile.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr %e syncscope("cluster") monotonic, align 4

  ; CHECK: ld.volatile.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr %e syncscope("cluster") monotonic, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.volatile.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr %e syncscope("cluster") monotonic, align 8

  ret void
}

; CHECK-LABEL: generic_acq_rel_cluster
define void @generic_acq_rel_cluster(ptr %a, ptr %b, ptr %c, ptr %d, ptr %e) local_unnamed_addr {
  ; CHECK: ld.acquire.cluster.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr %a syncscope("cluster") acquire, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.release.cluster.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr %a syncscope("cluster") release, align 1

  ; CHECK: ld.acquire.cluster.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr %b syncscope("cluster") acquire, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.release.cluster.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr %b syncscope("cluster") release, align 2

  ; CHECK: ld.acquire.cluster.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr %c syncscope("cluster") acquire, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.release.cluster.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr %c syncscope("cluster") release, align 4

  ; CHECK: ld.acquire.cluster.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr %d syncscope("cluster") acquire, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.release.cluster.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr %d syncscope("cluster") release, align 8

  ; CHECK: ld.acquire.cluster.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr %e syncscope("cluster") acquire, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.release.cluster.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr %e syncscope("cluster") release, align 4

  ; CHECK: ld.acquire.cluster.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr %e syncscope("cluster") acquire, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.release.cluster.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr %e syncscope("cluster") release, align 8

  ret void
}

; CHECK-LABEL: generic_acq_rel_volatile_cluster
define void @generic_acq_rel_volatile_cluster(ptr %a, ptr %b, ptr %c, ptr %d, ptr %e) local_unnamed_addr {
  ; CHECK: ld.acquire.sys.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr %a syncscope("cluster") acquire, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.release.sys.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr %a syncscope("cluster") release, align 1

  ; CHECK: ld.acquire.sys.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr %b syncscope("cluster") acquire, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.release.sys.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr %b syncscope("cluster") release, align 2

  ; CHECK: ld.acquire.sys.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr %c syncscope("cluster") acquire, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.release.sys.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr %c syncscope("cluster") release, align 4

  ; CHECK: ld.acquire.sys.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr %d syncscope("cluster") acquire, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.release.sys.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr %d syncscope("cluster") release, align 8

  ; CHECK: ld.acquire.sys.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr %e syncscope("cluster") acquire, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.release.sys.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr %e syncscope("cluster") release, align 4

  ; CHECK: ld.acquire.sys.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr %e syncscope("cluster") acquire, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.release.sys.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr %e syncscope("cluster") release, align 8

  ret void
}

; CHECK-LABEL: generic_sc_cluster
define void @generic_sc_cluster(ptr %a, ptr %b, ptr %c, ptr %d, ptr %e) local_unnamed_addr {
  ; CHECK: fence.sc.cluster
  ; CHECK: ld.acquire.cluster.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr %a syncscope("cluster") seq_cst, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: fence.sc.cluster
  ; CHECK: st.release.cluster.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr %a syncscope("cluster") seq_cst, align 1

  ; CHECK: fence.sc.cluster
  ; CHECK: ld.acquire.cluster.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr %b syncscope("cluster") seq_cst, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: fence.sc.cluster
  ; CHECK: st.release.cluster.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr %b syncscope("cluster") seq_cst, align 2

  ; CHECK: fence.sc.cluster
  ; CHECK: ld.acquire.cluster.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr %c syncscope("cluster") seq_cst, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: fence.sc.cluster
  ; CHECK: st.release.cluster.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr %c syncscope("cluster") seq_cst, align 4

  ; CHECK: fence.sc.cluster
  ; CHECK: ld.acquire.cluster.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr %d syncscope("cluster") seq_cst, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: fence.sc.cluster
  ; CHECK: st.release.cluster.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr %d syncscope("cluster") seq_cst, align 8

  ; CHECK: fence.sc.cluster
  ; CHECK: ld.acquire.cluster.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr %e syncscope("cluster") seq_cst, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: fence.sc.cluster
  ; CHECK: st.release.cluster.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr %e syncscope("cluster") seq_cst, align 4

  ; CHECK: fence.sc.cluster
  ; CHECK: ld.acquire.cluster.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr %e syncscope("cluster") seq_cst, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: fence.sc.cluster
  ; CHECK: st.release.cluster.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr %e syncscope("cluster") seq_cst, align 8

  ret void
}

; CHECK-LABEL: generic_sc_volatile_cluster
define void @generic_sc_volatile_cluster(ptr %a, ptr %b, ptr %c, ptr %d, ptr %e) local_unnamed_addr {
  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr %a syncscope("cluster") seq_cst, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr %a syncscope("cluster") seq_cst, align 1

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr %b syncscope("cluster") seq_cst, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr %b syncscope("cluster") seq_cst, align 2

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr %c syncscope("cluster") seq_cst, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr %c syncscope("cluster") seq_cst, align 4

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr %d syncscope("cluster") seq_cst, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr %d syncscope("cluster") seq_cst, align 8

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr %e syncscope("cluster") seq_cst, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr %e syncscope("cluster") seq_cst, align 4

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr %e syncscope("cluster") seq_cst, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr %e syncscope("cluster") seq_cst, align 8

  ret void
}

;; global statespace

; CHECK-LABEL: global_unordered_cluster
define void @global_unordered_cluster(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %e) local_unnamed_addr {
  ; CHECK: ld.relaxed.cluster.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(1) %a syncscope("cluster") unordered, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.relaxed.cluster.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(1) %a syncscope("cluster") unordered, align 1

  ; CHECK: ld.relaxed.cluster.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(1) %b syncscope("cluster") unordered, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.relaxed.cluster.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(1) %b syncscope("cluster") unordered, align 2

  ; CHECK: ld.relaxed.cluster.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(1) %c syncscope("cluster") unordered, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.relaxed.cluster.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(1) %c syncscope("cluster") unordered, align 4

  ; CHECK: ld.relaxed.cluster.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(1) %d syncscope("cluster") unordered, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.relaxed.cluster.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(1) %d syncscope("cluster") unordered, align 8

  ; CHECK: ld.relaxed.cluster.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(1) %e syncscope("cluster") unordered, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.relaxed.cluster.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(1) %e syncscope("cluster") unordered, align 4

  ; CHECK: ld.relaxed.cluster.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(1) %e syncscope("cluster") unordered, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.relaxed.cluster.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(1) %e syncscope("cluster") unordered, align 8

  ret void
}

; CHECK-LABEL: global_unordered_volatile_cluster
define void @global_unordered_volatile_cluster(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %e) local_unnamed_addr {
  ; CHECK: ld.volatile.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(1) %a syncscope("cluster") unordered, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.volatile.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(1) %a syncscope("cluster") unordered, align 1

  ; CHECK: ld.volatile.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(1) %b syncscope("cluster") unordered, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.volatile.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(1) %b syncscope("cluster") unordered, align 2

  ; CHECK: ld.volatile.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(1) %c syncscope("cluster") unordered, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.volatile.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(1) %c syncscope("cluster") unordered, align 4

  ; CHECK: ld.volatile.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(1) %d syncscope("cluster") unordered, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.volatile.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(1) %d syncscope("cluster") unordered, align 8

  ; CHECK: ld.volatile.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(1) %e syncscope("cluster") unordered, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.volatile.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(1) %e syncscope("cluster") unordered, align 4

  ; CHECK: ld.volatile.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(1) %e syncscope("cluster") unordered, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.volatile.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(1) %e syncscope("cluster") unordered, align 8

  ret void
}

; CHECK-LABEL: global_monotonic_cluster
define void @global_monotonic_cluster(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %e) local_unnamed_addr {
  ; CHECK: ld.relaxed.cluster.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(1) %a syncscope("cluster") monotonic, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.relaxed.cluster.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(1) %a syncscope("cluster") monotonic, align 1

  ; CHECK: ld.relaxed.cluster.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(1) %b syncscope("cluster") monotonic, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.relaxed.cluster.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(1) %b syncscope("cluster") monotonic, align 2

  ; CHECK: ld.relaxed.cluster.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(1) %c syncscope("cluster") monotonic, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.relaxed.cluster.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(1) %c syncscope("cluster") monotonic, align 4

  ; CHECK: ld.relaxed.cluster.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(1) %d syncscope("cluster") monotonic, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.relaxed.cluster.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(1) %d syncscope("cluster") monotonic, align 8

  ; CHECK: ld.relaxed.cluster.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(1) %e syncscope("cluster") monotonic, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.relaxed.cluster.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(1) %e syncscope("cluster") monotonic, align 4

  ; CHECK: ld.relaxed.cluster.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(1) %e syncscope("cluster") monotonic, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.relaxed.cluster.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(1) %e syncscope("cluster") monotonic, align 8

  ret void
}

; CHECK-LABEL: global_monotonic_volatile_cluster
define void @global_monotonic_volatile_cluster(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %e) local_unnamed_addr {
  ; CHECK: ld.volatile.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(1) %a syncscope("cluster") monotonic, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.volatile.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(1) %a syncscope("cluster") monotonic, align 1

  ; CHECK: ld.volatile.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(1) %b syncscope("cluster") monotonic, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.volatile.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(1) %b syncscope("cluster") monotonic, align 2

  ; CHECK: ld.volatile.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(1) %c syncscope("cluster") monotonic, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.volatile.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(1) %c syncscope("cluster") monotonic, align 4

  ; CHECK: ld.volatile.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(1) %d syncscope("cluster") monotonic, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.volatile.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(1) %d syncscope("cluster") monotonic, align 8

  ; CHECK: ld.volatile.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(1) %e syncscope("cluster") monotonic, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.volatile.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(1) %e syncscope("cluster") monotonic, align 4

  ; CHECK: ld.volatile.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(1) %e syncscope("cluster") monotonic, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.volatile.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(1) %e syncscope("cluster") monotonic, align 8

  ret void
}

; CHECK-LABEL: global_acq_rel_cluster
define void @global_acq_rel_cluster(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %e) local_unnamed_addr {
  ; CHECK: ld.acquire.cluster.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(1) %a syncscope("cluster") acquire, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.release.cluster.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(1) %a syncscope("cluster") release, align 1

  ; CHECK: ld.acquire.cluster.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(1) %b syncscope("cluster") acquire, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.release.cluster.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(1) %b syncscope("cluster") release, align 2

  ; CHECK: ld.acquire.cluster.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(1) %c syncscope("cluster") acquire, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.release.cluster.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(1) %c syncscope("cluster") release, align 4

  ; CHECK: ld.acquire.cluster.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(1) %d syncscope("cluster") acquire, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.release.cluster.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(1) %d syncscope("cluster") release, align 8

  ; CHECK: ld.acquire.cluster.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(1) %e syncscope("cluster") acquire, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.release.cluster.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(1) %e syncscope("cluster") release, align 4

  ; CHECK: ld.acquire.cluster.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(1) %e syncscope("cluster") acquire, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.release.cluster.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(1) %e syncscope("cluster") release, align 8

  ret void
}

; CHECK-LABEL: global_acq_rel_volatile_cluster
define void @global_acq_rel_volatile_cluster(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %e) local_unnamed_addr {
  ; CHECK: ld.acquire.sys.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(1) %a syncscope("cluster") acquire, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.release.sys.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(1) %a syncscope("cluster") release, align 1

  ; CHECK: ld.acquire.sys.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(1) %b syncscope("cluster") acquire, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.release.sys.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(1) %b syncscope("cluster") release, align 2

  ; CHECK: ld.acquire.sys.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(1) %c syncscope("cluster") acquire, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.release.sys.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(1) %c syncscope("cluster") release, align 4

  ; CHECK: ld.acquire.sys.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(1) %d syncscope("cluster") acquire, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.release.sys.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(1) %d syncscope("cluster") release, align 8

  ; CHECK: ld.acquire.sys.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(1) %e syncscope("cluster") acquire, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.release.sys.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(1) %e syncscope("cluster") release, align 4

  ; CHECK: ld.acquire.sys.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(1) %e syncscope("cluster") acquire, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.release.sys.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(1) %e syncscope("cluster") release, align 8

  ret void
}

; CHECK-LABEL: global_seq_cst_cluster
define void @global_seq_cst_cluster(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %e) local_unnamed_addr {
  ; CHECK: fence.sc.cluster
  ; CHECK: ld.acquire.cluster.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(1) %a syncscope("cluster") seq_cst, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: fence.sc.cluster
  ; CHECK: st.release.cluster.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(1) %a syncscope("cluster") seq_cst, align 1

  ; CHECK: fence.sc.cluster
  ; CHECK: ld.acquire.cluster.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(1) %b syncscope("cluster") seq_cst, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: fence.sc.cluster
  ; CHECK: st.release.cluster.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(1) %b syncscope("cluster") seq_cst, align 2

  ; CHECK: fence.sc.cluster
  ; CHECK: ld.acquire.cluster.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(1) %c syncscope("cluster") seq_cst, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: fence.sc.cluster
  ; CHECK: st.release.cluster.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(1) %c syncscope("cluster") seq_cst, align 4

  ; CHECK: fence.sc.cluster
  ; CHECK: ld.acquire.cluster.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(1) %d syncscope("cluster") seq_cst, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: fence.sc.cluster
  ; CHECK: st.release.cluster.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(1) %d syncscope("cluster") seq_cst, align 8

  ; CHECK: fence.sc.cluster
  ; CHECK: ld.acquire.cluster.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(1) %e syncscope("cluster") seq_cst, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: fence.sc.cluster
  ; CHECK: st.release.cluster.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(1) %e syncscope("cluster") seq_cst, align 4

  ; CHECK: fence.sc.cluster
  ; CHECK: ld.acquire.cluster.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(1) %e syncscope("cluster") seq_cst, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: fence.sc.cluster
  ; CHECK: st.release.cluster.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(1) %e syncscope("cluster") seq_cst, align 8

  ret void
}

; CHECK-LABEL: global_seq_cst_volatile_cluster
define void @global_seq_cst_volatile_cluster(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %e) local_unnamed_addr {
  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(1) %a syncscope("cluster") seq_cst, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(1) %a syncscope("cluster") seq_cst, align 1

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(1) %b syncscope("cluster") seq_cst, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(1) %b syncscope("cluster") seq_cst, align 2

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(1) %c syncscope("cluster") seq_cst, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(1) %c syncscope("cluster") seq_cst, align 4

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(1) %d syncscope("cluster") seq_cst, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(1) %d syncscope("cluster") seq_cst, align 8

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(1) %e syncscope("cluster") seq_cst, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(1) %e syncscope("cluster") seq_cst, align 4

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(1) %e syncscope("cluster") seq_cst, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(1) %e syncscope("cluster") seq_cst, align 8

  ret void
}

;; shared

; CHECK-LABEL: shared_unordered_cluster
define void @shared_unordered_cluster(ptr addrspace(3) %a, ptr addrspace(3) %b, ptr addrspace(3) %c, ptr addrspace(3) %d, ptr addrspace(3) %e) local_unnamed_addr {
  ; CHECK: ld.relaxed.cluster.shared.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(3) %a syncscope("cluster") unordered, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.relaxed.cluster.shared.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(3) %a syncscope("cluster") unordered, align 1

  ; CHECK: ld.relaxed.cluster.shared.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(3) %b syncscope("cluster") unordered, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.relaxed.cluster.shared.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(3) %b syncscope("cluster") unordered, align 2

  ; CHECK: ld.relaxed.cluster.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(3) %c syncscope("cluster") unordered, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.relaxed.cluster.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(3) %c syncscope("cluster") unordered, align 4

  ; CHECK: ld.relaxed.cluster.shared.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(3) %d syncscope("cluster") unordered, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.relaxed.cluster.shared.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(3) %d syncscope("cluster") unordered, align 8

  ; CHECK: ld.relaxed.cluster.shared.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(3) %e syncscope("cluster") unordered, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.relaxed.cluster.shared.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(3) %e syncscope("cluster") unordered, align 4

  ; CHECK: ld.relaxed.cluster.shared.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(3) %e syncscope("cluster") unordered, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.relaxed.cluster.shared.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(3) %e syncscope("cluster") unordered, align 8

  ret void
}

; CHECK-LABEL: shared_unordered_volatile_cluster
define void @shared_unordered_volatile_cluster(ptr addrspace(3) %a, ptr addrspace(3) %b, ptr addrspace(3) %c, ptr addrspace(3) %d, ptr addrspace(3) %e) local_unnamed_addr {
  ; CHECK: ld.volatile.shared.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(3) %a syncscope("cluster") unordered, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.volatile.shared.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(3) %a syncscope("cluster") unordered, align 1

  ; CHECK: ld.volatile.shared.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(3) %b syncscope("cluster") unordered, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.volatile.shared.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(3) %b syncscope("cluster") unordered, align 2

  ; CHECK: ld.volatile.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(3) %c syncscope("cluster") unordered, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.volatile.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(3) %c syncscope("cluster") unordered, align 4

  ; CHECK: ld.volatile.shared.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(3) %d syncscope("cluster") unordered, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.volatile.shared.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(3) %d syncscope("cluster") unordered, align 8

  ; CHECK: ld.volatile.shared.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(3) %e syncscope("cluster") unordered, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.volatile.shared.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(3) %e syncscope("cluster") unordered, align 4

  ; CHECK: ld.volatile.shared.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(3) %e syncscope("cluster") unordered, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.volatile.shared.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(3) %e syncscope("cluster") unordered, align 8

  ret void
}

; CHECK-LABEL: shared_monotonic_cluster
define void @shared_monotonic_cluster(ptr addrspace(3) %a, ptr addrspace(3) %b, ptr addrspace(3) %c, ptr addrspace(3) %d, ptr addrspace(3) %e) local_unnamed_addr {
  ; CHECK: ld.relaxed.cluster.shared.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(3) %a syncscope("cluster") monotonic, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.relaxed.cluster.shared.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(3) %a syncscope("cluster") monotonic, align 1

  ; CHECK: ld.relaxed.cluster.shared.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(3) %b syncscope("cluster") monotonic, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.relaxed.cluster.shared.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(3) %b syncscope("cluster") monotonic, align 2

  ; CHECK: ld.relaxed.cluster.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(3) %c syncscope("cluster") monotonic, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.relaxed.cluster.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(3) %c syncscope("cluster") monotonic, align 4

  ; CHECK: ld.relaxed.cluster.shared.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(3) %d syncscope("cluster") monotonic, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.relaxed.cluster.shared.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(3) %d syncscope("cluster") monotonic, align 8

  ; CHECK: ld.relaxed.cluster.shared.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(3) %e syncscope("cluster") monotonic, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.relaxed.cluster.shared.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(3) %e syncscope("cluster") monotonic, align 4

  ; CHECK: ld.relaxed.cluster.shared.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(3) %e syncscope("cluster") monotonic, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.relaxed.cluster.shared.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(3) %e syncscope("cluster") monotonic, align 8

  ret void
}

; CHECK-LABEL: shared_monotonic_volatile_cluster
define void @shared_monotonic_volatile_cluster(ptr addrspace(3) %a, ptr addrspace(3) %b, ptr addrspace(3) %c, ptr addrspace(3) %d, ptr addrspace(3) %e) local_unnamed_addr {
  ; CHECK: ld.volatile.shared.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(3) %a syncscope("cluster") monotonic, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.volatile.shared.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(3) %a syncscope("cluster") monotonic, align 1

  ; CHECK: ld.volatile.shared.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(3) %b syncscope("cluster") monotonic, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.volatile.shared.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(3) %b syncscope("cluster") monotonic, align 2

  ; CHECK: ld.volatile.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(3) %c syncscope("cluster") monotonic, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.volatile.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(3) %c syncscope("cluster") monotonic, align 4

  ; CHECK: ld.volatile.shared.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(3) %d syncscope("cluster") monotonic, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.volatile.shared.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(3) %d syncscope("cluster") monotonic, align 8

  ; CHECK: ld.volatile.shared.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(3) %e syncscope("cluster") monotonic, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.volatile.shared.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(3) %e syncscope("cluster") monotonic, align 4

  ; CHECK: ld.volatile.shared.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(3) %e syncscope("cluster") monotonic, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.volatile.shared.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(3) %e syncscope("cluster") monotonic, align 8

  ret void
}

; CHECK-LABEL: shared_acq_rel_cluster
define void @shared_acq_rel_cluster(ptr addrspace(3) %a, ptr addrspace(3) %b, ptr addrspace(3) %c, ptr addrspace(3) %d, ptr addrspace(3) %e) local_unnamed_addr {
  ; CHECK: ld.acquire.cluster.shared.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(3) %a syncscope("cluster") acquire, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.release.cluster.shared.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(3) %a syncscope("cluster") release, align 1

  ; CHECK: ld.acquire.cluster.shared.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(3) %b syncscope("cluster") acquire, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.release.cluster.shared.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(3) %b syncscope("cluster") release, align 2

  ; CHECK: ld.acquire.cluster.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(3) %c syncscope("cluster") acquire, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.release.cluster.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(3) %c syncscope("cluster") release, align 4

  ; CHECK: ld.acquire.cluster.shared.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(3) %d syncscope("cluster") acquire, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.release.cluster.shared.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(3) %d syncscope("cluster") release, align 8

  ; CHECK: ld.acquire.cluster.shared.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(3) %e syncscope("cluster") acquire, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.release.cluster.shared.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(3) %e syncscope("cluster") release, align 4

  ; CHECK: ld.acquire.cluster.shared.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(3) %e syncscope("cluster") acquire, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.release.cluster.shared.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(3) %e syncscope("cluster") release, align 8

  ret void
}

; CHECK-LABEL: shared_acq_rel_volatile_cluster
define void @shared_acq_rel_volatile_cluster(ptr addrspace(3) %a, ptr addrspace(3) %b, ptr addrspace(3) %c, ptr addrspace(3) %d, ptr addrspace(3) %e) local_unnamed_addr {
  ; CHECK: ld.acquire.sys.shared.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(3) %a syncscope("cluster") acquire, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.release.sys.shared.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(3) %a syncscope("cluster") release, align 1

  ; CHECK: ld.acquire.sys.shared.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(3) %b syncscope("cluster") acquire, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.release.sys.shared.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(3) %b syncscope("cluster") release, align 2

  ; CHECK: ld.acquire.sys.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(3) %c syncscope("cluster") acquire, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.release.sys.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(3) %c syncscope("cluster") release, align 4

  ; CHECK: ld.acquire.sys.shared.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(3) %d syncscope("cluster") acquire, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.release.sys.shared.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(3) %d syncscope("cluster") release, align 8

  ; CHECK: ld.acquire.sys.shared.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(3) %e syncscope("cluster") acquire, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.release.sys.shared.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(3) %e syncscope("cluster") release, align 4

  ; CHECK: ld.acquire.sys.shared.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(3) %e syncscope("cluster") acquire, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.release.sys.shared.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(3) %e syncscope("cluster") release, align 8

  ret void
}

; CHECK-LABEL: shared_seq_cst_cluster
define void @shared_seq_cst_cluster(ptr addrspace(3) %a, ptr addrspace(3) %b, ptr addrspace(3) %c, ptr addrspace(3) %d, ptr addrspace(3) %e) local_unnamed_addr {
  ; CHECK: fence.sc.cluster
  ; CHECK: ld.acquire.cluster.shared.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(3) %a syncscope("cluster") seq_cst, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: fence.sc.cluster
  ; CHECK: st.release.cluster.shared.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(3) %a syncscope("cluster") seq_cst, align 1

  ; CHECK: fence.sc.cluster
  ; CHECK: ld.acquire.cluster.shared.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(3) %b syncscope("cluster") seq_cst, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: fence.sc.cluster
  ; CHECK: st.release.cluster.shared.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(3) %b syncscope("cluster") seq_cst, align 2

  ; CHECK: fence.sc.cluster
  ; CHECK: ld.acquire.cluster.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(3) %c syncscope("cluster") seq_cst, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: fence.sc.cluster
  ; CHECK: st.release.cluster.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(3) %c syncscope("cluster") seq_cst, align 4

  ; CHECK: fence.sc.cluster
  ; CHECK: ld.acquire.cluster.shared.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(3) %d syncscope("cluster") seq_cst, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: fence.sc.cluster
  ; CHECK: st.release.cluster.shared.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(3) %d syncscope("cluster") seq_cst, align 8

  ; CHECK: fence.sc.cluster
  ; CHECK: ld.acquire.cluster.shared.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(3) %e syncscope("cluster") seq_cst, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: fence.sc.cluster
  ; CHECK: st.release.cluster.shared.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(3) %e syncscope("cluster") seq_cst, align 4

  ; CHECK: fence.sc.cluster
  ; CHECK: ld.acquire.cluster.shared.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(3) %e syncscope("cluster") seq_cst, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: fence.sc.cluster
  ; CHECK: st.release.cluster.shared.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(3) %e syncscope("cluster") seq_cst, align 8

  ret void
}

; CHECK-LABEL: shared_seq_cst_volatile_cluster
define void @shared_seq_cst_volatile_cluster(ptr addrspace(3) %a, ptr addrspace(3) %b, ptr addrspace(3) %c, ptr addrspace(3) %d, ptr addrspace(3) %e) local_unnamed_addr {
  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.shared.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(3) %a syncscope("cluster") seq_cst, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.shared.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(3) %a syncscope("cluster") seq_cst, align 1

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.shared.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(3) %b syncscope("cluster") seq_cst, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.shared.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(3) %b syncscope("cluster") seq_cst, align 2

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(3) %c syncscope("cluster") seq_cst, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(3) %c syncscope("cluster") seq_cst, align 4

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.shared.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(3) %d syncscope("cluster") seq_cst, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.shared.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(3) %d syncscope("cluster") seq_cst, align 8

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.shared.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(3) %e syncscope("cluster") seq_cst, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.shared.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(3) %e syncscope("cluster") seq_cst, align 4

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.shared.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(3) %e syncscope("cluster") seq_cst, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.shared.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(3) %e syncscope("cluster") seq_cst, align 8

  ret void
}

;; local statespace

; CHECK-LABEL: local_unordered_cluster
define void @local_unordered_cluster(ptr addrspace(5) %a, ptr addrspace(5) %b, ptr addrspace(5) %c, ptr addrspace(5) %d, ptr addrspace(5) %e) local_unnamed_addr {
  ; CHECK: ld.local.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(5) %a syncscope("cluster") unordered, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.local.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(5) %a syncscope("cluster") unordered, align 1

  ; CHECK: ld.local.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(5) %b syncscope("cluster") unordered, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.local.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(5) %b syncscope("cluster") unordered, align 2

  ; CHECK: ld.local.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(5) %c syncscope("cluster") unordered, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.local.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(5) %c syncscope("cluster") unordered, align 4

  ; CHECK: ld.local.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(5) %d syncscope("cluster") unordered, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.local.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(5) %d syncscope("cluster") unordered, align 8

  ; CHECK: ld.local.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(5) %e syncscope("cluster") unordered, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.local.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(5) %e syncscope("cluster") unordered, align 4

  ; CHECK: ld.local.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(5) %e syncscope("cluster") unordered, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.local.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(5) %e syncscope("cluster") unordered, align 8

  ret void
}

; CHECK-LABEL: local_unordered_volatile_cluster
define void @local_unordered_volatile_cluster(ptr addrspace(5) %a, ptr addrspace(5) %b, ptr addrspace(5) %c, ptr addrspace(5) %d, ptr addrspace(5) %e) local_unnamed_addr {
  ; CHECK: ld.local.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(5) %a syncscope("cluster") unordered, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.local.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(5) %a syncscope("cluster") unordered, align 1

  ; CHECK: ld.local.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(5) %b syncscope("cluster") unordered, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.local.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(5) %b syncscope("cluster") unordered, align 2

  ; CHECK: ld.local.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(5) %c syncscope("cluster") unordered, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.local.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(5) %c syncscope("cluster") unordered, align 4

  ; CHECK: ld.local.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(5) %d syncscope("cluster") unordered, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.local.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(5) %d syncscope("cluster") unordered, align 8

  ; CHECK: ld.local.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(5) %e syncscope("cluster") unordered, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.local.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(5) %e syncscope("cluster") unordered, align 4

  ; CHECK: ld.local.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(5) %e syncscope("cluster") unordered, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.local.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(5) %e syncscope("cluster") unordered, align 8

  ret void
}

; CHECK-LABEL: local_monotonic_cluster
define void @local_monotonic_cluster(ptr addrspace(5) %a, ptr addrspace(5) %b, ptr addrspace(5) %c, ptr addrspace(5) %d, ptr addrspace(5) %e) local_unnamed_addr {
  ; CHECK: ld.local.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(5) %a syncscope("cluster") monotonic, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.local.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(5) %a syncscope("cluster") monotonic, align 1

  ; CHECK: ld.local.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(5) %b syncscope("cluster") monotonic, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.local.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(5) %b syncscope("cluster") monotonic, align 2

  ; CHECK: ld.local.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(5) %c syncscope("cluster") monotonic, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.local.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(5) %c syncscope("cluster") monotonic, align 4

  ; CHECK: ld.local.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(5) %d syncscope("cluster") monotonic, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.local.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(5) %d syncscope("cluster") monotonic, align 8

  ; CHECK: ld.local.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(5) %e syncscope("cluster") monotonic, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.local.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(5) %e syncscope("cluster") monotonic, align 4

  ; CHECK: ld.local.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(5) %e syncscope("cluster") monotonic, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.local.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(5) %e syncscope("cluster") monotonic, align 8

  ret void
}

; CHECK-LABEL: local_monotonic_volatile_cluster
define void @local_monotonic_volatile_cluster(ptr addrspace(5) %a, ptr addrspace(5) %b, ptr addrspace(5) %c, ptr addrspace(5) %d, ptr addrspace(5) %e) local_unnamed_addr {
  ; CHECK: ld.local.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(5) %a syncscope("cluster") monotonic, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.local.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(5) %a syncscope("cluster") monotonic, align 1

  ; CHECK: ld.local.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(5) %b syncscope("cluster") monotonic, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.local.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(5) %b syncscope("cluster") monotonic, align 2

  ; CHECK: ld.local.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(5) %c syncscope("cluster") monotonic, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.local.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(5) %c syncscope("cluster") monotonic, align 4

  ; CHECK: ld.local.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(5) %d syncscope("cluster") monotonic, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.local.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(5) %d syncscope("cluster") monotonic, align 8

  ; CHECK: ld.local.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(5) %e syncscope("cluster") monotonic, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.local.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(5) %e syncscope("cluster") monotonic, align 4

  ; CHECK: ld.local.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(5) %e syncscope("cluster") monotonic, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.local.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(5) %e syncscope("cluster") monotonic, align 8

  ret void
}

; CHECK-LABEL: local_acq_rel_cluster
define void @local_acq_rel_cluster(ptr addrspace(5) %a, ptr addrspace(5) %b, ptr addrspace(5) %c, ptr addrspace(5) %d, ptr addrspace(5) %e) local_unnamed_addr {
  ; CHECK: ld.local.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(5) %a syncscope("cluster") acquire, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.local.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(5) %a syncscope("cluster") release, align 1

  ; CHECK: ld.local.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(5) %b syncscope("cluster") acquire, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.local.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(5) %b syncscope("cluster") release, align 2

  ; CHECK: ld.local.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(5) %c syncscope("cluster") acquire, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.local.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(5) %c syncscope("cluster") release, align 4

  ; CHECK: ld.local.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(5) %d syncscope("cluster") acquire, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.local.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(5) %d syncscope("cluster") release, align 8

  ; CHECK: ld.local.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(5) %e syncscope("cluster") acquire, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.local.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(5) %e syncscope("cluster") release, align 4

  ; CHECK: ld.local.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(5) %e syncscope("cluster") acquire, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.local.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(5) %e syncscope("cluster") release, align 8

  ret void
}

; CHECK-LABEL: local_acq_rel_volatile_cluster
define void @local_acq_rel_volatile_cluster(ptr addrspace(5) %a, ptr addrspace(5) %b, ptr addrspace(5) %c, ptr addrspace(5) %d, ptr addrspace(5) %e) local_unnamed_addr {
  ; CHECK: ld.local.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(5) %a syncscope("cluster") acquire, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.local.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(5) %a syncscope("cluster") release, align 1

  ; CHECK: ld.local.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(5) %b syncscope("cluster") acquire, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.local.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(5) %b syncscope("cluster") release, align 2

  ; CHECK: ld.local.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(5) %c syncscope("cluster") acquire, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.local.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(5) %c syncscope("cluster") release, align 4

  ; CHECK: ld.local.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(5) %d syncscope("cluster") acquire, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.local.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(5) %d syncscope("cluster") release, align 8

  ; CHECK: ld.local.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(5) %e syncscope("cluster") acquire, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.local.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(5) %e syncscope("cluster") release, align 4

  ; CHECK: ld.local.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(5) %e syncscope("cluster") acquire, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.local.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(5) %e syncscope("cluster") release, align 8

  ret void
}

; CHECK-LABEL: local_seq_cst_cluster
define void @local_seq_cst_cluster(ptr addrspace(5) %a, ptr addrspace(5) %b, ptr addrspace(5) %c, ptr addrspace(5) %d, ptr addrspace(5) %e) local_unnamed_addr {
  ; CHECK: ld.local.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(5) %a syncscope("cluster") seq_cst, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.local.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(5) %a syncscope("cluster") seq_cst, align 1

  ; CHECK: ld.local.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(5) %b syncscope("cluster") seq_cst, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.local.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(5) %b syncscope("cluster") seq_cst, align 2

  ; CHECK: ld.local.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(5) %c syncscope("cluster") seq_cst, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.local.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(5) %c syncscope("cluster") seq_cst, align 4

  ; CHECK: ld.local.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(5) %d syncscope("cluster") seq_cst, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.local.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(5) %d syncscope("cluster") seq_cst, align 8

  ; CHECK: ld.local.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(5) %e syncscope("cluster") seq_cst, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.local.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(5) %e syncscope("cluster") seq_cst, align 4

  ; CHECK: ld.local.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(5) %e syncscope("cluster") seq_cst, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.local.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(5) %e syncscope("cluster") seq_cst, align 8

  ret void
}

; CHECK-LABEL: local_seq_cst_volatile_cluster
define void @local_seq_cst_volatile_cluster(ptr addrspace(5) %a, ptr addrspace(5) %b, ptr addrspace(5) %c, ptr addrspace(5) %d, ptr addrspace(5) %e) local_unnamed_addr {
  ; CHECK: ld.local.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(5) %a syncscope("cluster") seq_cst, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.local.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(5) %a syncscope("cluster") seq_cst, align 1

  ; CHECK: ld.local.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(5) %b syncscope("cluster") seq_cst, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.local.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(5) %b syncscope("cluster") seq_cst, align 2

  ; CHECK: ld.local.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(5) %c syncscope("cluster") seq_cst, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.local.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(5) %c syncscope("cluster") seq_cst, align 4

  ; CHECK: ld.local.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(5) %d syncscope("cluster") seq_cst, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.local.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(5) %d syncscope("cluster") seq_cst, align 8

  ; CHECK: ld.local.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(5) %e syncscope("cluster") seq_cst, align 4
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.local.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(5) %e syncscope("cluster") seq_cst, align 4

  ; CHECK: ld.local.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(5) %e syncscope("cluster") seq_cst, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.local.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(5) %e syncscope("cluster") seq_cst, align 8

  ret void
}
