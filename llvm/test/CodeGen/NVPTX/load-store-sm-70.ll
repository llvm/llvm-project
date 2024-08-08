; RUN: llc < %s -march=nvptx64 -mcpu=sm_70 -mattr=+ptx82 | FileCheck %s
; RUN: %if ptxas-12.2 %{ llc < %s -march=nvptx64 -mcpu=sm_70 -mattr=+ptx82 | %ptxas-verify -arch=sm_70 %}

;; generic statespace

; CHECK-LABEL: generic_acq_rel
define void @generic_acq_rel(ptr %a, ptr %b, ptr %c, ptr %d, ptr %e) local_unnamed_addr {
  ; CHECK: ld.acquire.sys.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr %a acquire, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.release.sys.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr %a release, align 1

  ; CHECK: ld.acquire.sys.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr %b acquire, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.release.sys.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr %b release, align 2

  ; CHECK: ld.acquire.sys.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr %c acquire, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.release.sys.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr %c release, align 4

  ; CHECK: ld.acquire.sys.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr %d acquire, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.release.sys.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr %d release, align 8

  ; CHECK: ld.acquire.sys.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr %e acquire, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.release.sys.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr %e release, align 4

  ; CHECK: ld.acquire.sys.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr %e acquire, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.release.sys.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr %e release, align 8

  ret void
}

; CHECK-LABEL: generic_acq_rel_volatile
define void @generic_acq_rel_volatile(ptr %a, ptr %b, ptr %c, ptr %d, ptr %e) local_unnamed_addr {
  ; CHECK: ld.acquire.sys.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr %a acquire, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.release.sys.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr %a release, align 1

  ; CHECK: ld.acquire.sys.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr %b acquire, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.release.sys.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr %b release, align 2

  ; CHECK: ld.acquire.sys.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr %c acquire, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.release.sys.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr %c release, align 4

  ; CHECK: ld.acquire.sys.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr %d acquire, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.release.sys.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr %d release, align 8

  ; CHECK: ld.acquire.sys.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr %e acquire, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.release.sys.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr %e release, align 4

  ; CHECK: ld.acquire.sys.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr %e acquire, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.release.sys.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr %e release, align 8

  ret void
}

; CHECK-LABEL: generic_sc
define void @generic_sc(ptr %a, ptr %b, ptr %c, ptr %d, ptr %e) local_unnamed_addr {
  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr %a seq_cst, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr %a seq_cst, align 1

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr %b seq_cst, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr %b seq_cst, align 2

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr %c seq_cst, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr %c seq_cst, align 4

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr %d seq_cst, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr %d seq_cst, align 8

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr %e seq_cst, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr %e seq_cst, align 4

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr %e seq_cst, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr %e seq_cst, align 8

  ret void
}

; CHECK-LABEL: generic_sc_volatile
define void @generic_sc_volatile(ptr %a, ptr %b, ptr %c, ptr %d, ptr %e) local_unnamed_addr {
  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr %a seq_cst, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr %a seq_cst, align 1

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr %b seq_cst, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr %b seq_cst, align 2

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr %c seq_cst, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr %c seq_cst, align 4

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr %d seq_cst, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr %d seq_cst, align 8

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr %e seq_cst, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr %e seq_cst, align 4

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr %e seq_cst, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr %e seq_cst, align 8

  ret void
}

;; global statespace

; CHECK-LABEL: global_acq_rel
define void @global_acq_rel(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %e) local_unnamed_addr {
  ; CHECK: ld.acquire.sys.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(1) %a acquire, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.release.sys.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(1) %a release, align 1

  ; CHECK: ld.acquire.sys.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(1) %b acquire, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.release.sys.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(1) %b release, align 2

  ; CHECK: ld.acquire.sys.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(1) %c acquire, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.release.sys.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(1) %c release, align 4

  ; CHECK: ld.acquire.sys.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(1) %d acquire, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.release.sys.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(1) %d release, align 8

  ; CHECK: ld.acquire.sys.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(1) %e acquire, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.release.sys.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(1) %e release, align 4

  ; CHECK: ld.acquire.sys.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(1) %e acquire, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.release.sys.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(1) %e release, align 8

  ret void
}

; CHECK-LABEL: global_acq_rel_volatile
define void @global_acq_rel_volatile(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %e) local_unnamed_addr {
  ; CHECK: ld.acquire.sys.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(1) %a acquire, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.release.sys.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(1) %a release, align 1

  ; CHECK: ld.acquire.sys.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(1) %b acquire, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.release.sys.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(1) %b release, align 2

  ; CHECK: ld.acquire.sys.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(1) %c acquire, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.release.sys.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(1) %c release, align 4

  ; CHECK: ld.acquire.sys.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(1) %d acquire, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.release.sys.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(1) %d release, align 8

  ; CHECK: ld.acquire.sys.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(1) %e acquire, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.release.sys.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(1) %e release, align 4

  ; CHECK: ld.acquire.sys.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(1) %e acquire, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.release.sys.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(1) %e release, align 8

  ret void
}

; CHECK-LABEL: global_seq_cst
define void @global_seq_cst(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %e) local_unnamed_addr {
  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(1) %a seq_cst, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(1) %a seq_cst, align 1

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(1) %b seq_cst, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(1) %b seq_cst, align 2

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(1) %c seq_cst, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(1) %c seq_cst, align 4

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(1) %d seq_cst, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(1) %d seq_cst, align 8

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(1) %e seq_cst, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(1) %e seq_cst, align 4

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(1) %e seq_cst, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(1) %e seq_cst, align 8

  ret void
}

; CHECK-LABEL: global_seq_cst_volatile
define void @global_seq_cst_volatile(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %e) local_unnamed_addr {
  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(1) %a seq_cst, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(1) %a seq_cst, align 1

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(1) %b seq_cst, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(1) %b seq_cst, align 2

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(1) %c seq_cst, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(1) %c seq_cst, align 4

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(1) %d seq_cst, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(1) %d seq_cst, align 8

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(1) %e seq_cst, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(1) %e seq_cst, align 4

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(1) %e seq_cst, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(1) %e seq_cst, align 8

  ret void
}

;; shared statespace

; CHECK-LABEL: shared_acq_rel
define void @shared_acq_rel(ptr addrspace(3) %a, ptr addrspace(3) %b, ptr addrspace(3) %c, ptr addrspace(3) %d, ptr addrspace(3) %e) local_unnamed_addr {
  ; CHECK: ld.acquire.sys.shared.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(3) %a acquire, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.release.sys.shared.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(3) %a release, align 1

  ; CHECK: ld.acquire.sys.shared.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(3) %b acquire, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.release.sys.shared.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(3) %b release, align 2

  ; CHECK: ld.acquire.sys.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(3) %c acquire, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.release.sys.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(3) %c release, align 4

  ; CHECK: ld.acquire.sys.shared.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(3) %d acquire, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.release.sys.shared.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(3) %d release, align 8

  ; CHECK: ld.acquire.sys.shared.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(3) %e acquire, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.release.sys.shared.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(3) %e release, align 4

  ; CHECK: ld.acquire.sys.shared.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(3) %e acquire, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.release.sys.shared.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(3) %e release, align 8

  ret void
}

; CHECK-LABEL: shared_acq_rel_volatile
define void @shared_acq_rel_volatile(ptr addrspace(3) %a, ptr addrspace(3) %b, ptr addrspace(3) %c, ptr addrspace(3) %d, ptr addrspace(3) %e) local_unnamed_addr {
  ; CHECK: ld.acquire.sys.shared.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(3) %a acquire, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.release.sys.shared.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(3) %a release, align 1

  ; CHECK: ld.acquire.sys.shared.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(3) %b acquire, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.release.sys.shared.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(3) %b release, align 2

  ; CHECK: ld.acquire.sys.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(3) %c acquire, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.release.sys.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(3) %c release, align 4

  ; CHECK: ld.acquire.sys.shared.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(3) %d acquire, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.release.sys.shared.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(3) %d release, align 8

  ; CHECK: ld.acquire.sys.shared.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(3) %e acquire, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.release.sys.shared.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(3) %e release, align 4

  ; CHECK: ld.acquire.sys.shared.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(3) %e acquire, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.release.sys.shared.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(3) %e release, align 8

  ret void
}

; CHECK-LABEL: shared_seq_cst
define void @shared_seq_cst(ptr addrspace(3) %a, ptr addrspace(3) %b, ptr addrspace(3) %c, ptr addrspace(3) %d, ptr addrspace(3) %e) local_unnamed_addr {
  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.shared.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(3) %a seq_cst, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.shared.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(3) %a seq_cst, align 1

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.shared.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(3) %b seq_cst, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.shared.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(3) %b seq_cst, align 2

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(3) %c seq_cst, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(3) %c seq_cst, align 4

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.shared.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(3) %d seq_cst, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.shared.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(3) %d seq_cst, align 8

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.shared.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(3) %e seq_cst, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.shared.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(3) %e seq_cst, align 4

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.shared.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(3) %e seq_cst, align 8
  %f.add = fadd double %f.load, 1. 
  ; CHECK: fence.sc.sys 
  ; CHECK: st.release.sys.shared.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(3) %e seq_cst, align 8

  ret void
}

; CHECK-LABEL: shared_seq_cst_volatile
define void @shared_seq_cst_volatile(ptr addrspace(3) %a, ptr addrspace(3) %b, ptr addrspace(3) %c, ptr addrspace(3) %d, ptr addrspace(3) %e) local_unnamed_addr {
  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.shared.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(3) %a seq_cst, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.shared.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(3) %a seq_cst, align 1

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.shared.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(3) %b seq_cst, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.shared.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(3) %b seq_cst, align 2

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(3) %c seq_cst, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(3) %c seq_cst, align 4

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.shared.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(3) %d seq_cst, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.shared.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(3) %d seq_cst, align 8

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.shared.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(3) %e seq_cst, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.shared.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(3) %e seq_cst, align 4

  ; CHECK: fence.sc.sys
  ; CHECK: ld.acquire.sys.shared.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(3) %e seq_cst, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: fence.sc.sys
  ; CHECK: st.release.sys.shared.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(3) %e seq_cst, align 8

  ret void
}

;; local statespace

; CHECK-LABEL: local_acq_rel
define void @local_acq_rel(ptr addrspace(5) %a, ptr addrspace(5) %b, ptr addrspace(5) %c, ptr addrspace(5) %d, ptr addrspace(5) %e) local_unnamed_addr {
  ; TODO: generate PTX that preserves Concurrent Forward Progress
  ;       by using PTX atomic operations.

  ; CHECK: ld.local.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(5) %a acquire, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.local.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(5) %a release, align 1

  ; CHECK: ld.local.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(5) %b acquire, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.local.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(5) %b release, align 2

  ; CHECK: ld.local.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(5) %c acquire, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.local.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(5) %c release, align 4

  ; CHECK: ld.local.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(5) %d acquire, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.local.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(5) %d release, align 8

  ; CHECK: ld.local.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(5) %e acquire, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.local.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(5) %e release, align 4

  ; CHECK: ld.local.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(5) %e acquire, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.local.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(5) %e release, align 8

  ret void
}

; CHECK-LABEL: local_acq_rel_volatile
define void @local_acq_rel_volatile(ptr addrspace(5) %a, ptr addrspace(5) %b, ptr addrspace(5) %c, ptr addrspace(5) %d, ptr addrspace(5) %e) local_unnamed_addr {
  ; TODO: generate PTX that preserves Concurrent Forward Progress
  ;       by using PTX atomic operations.

  ; CHECK: ld.local.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(5) %a acquire, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.local.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(5) %a release, align 1

  ; CHECK: ld.local.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(5) %b acquire, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.local.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(5) %b release, align 2

  ; CHECK: ld.local.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(5) %c acquire, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.local.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(5) %c release, align 4

  ; CHECK: ld.local.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(5) %d acquire, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.local.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(5) %d release, align 8

  ; CHECK: ld.local.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(5) %e acquire, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.local.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(5) %e release, align 4

  ; CHECK: ld.local.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(5) %e acquire, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.local.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(5) %e release, align 8

  ret void
}

; CHECK-LABEL: local_seq_cst
define void @local_seq_cst(ptr addrspace(5) %a, ptr addrspace(5) %b, ptr addrspace(5) %c, ptr addrspace(5) %d, ptr addrspace(5) %e) local_unnamed_addr {
  ; TODO: generate PTX that preserves Concurrent Forward Progress
  ;       by using PTX atomic operations.

  ; CHECK: ld.local.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(5) %a seq_cst, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.local.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(5) %a seq_cst, align 1

  ; CHECK: ld.local.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(5) %b seq_cst, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.local.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(5) %b seq_cst, align 2

  ; CHECK: ld.local.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(5) %c seq_cst, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.local.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(5) %c seq_cst, align 4

  ; CHECK: ld.local.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(5) %d seq_cst, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.local.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(5) %d seq_cst, align 8

  ; CHECK: ld.local.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(5) %e seq_cst, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.local.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(5) %e seq_cst, align 4

  ; CHECK: ld.local.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(5) %e seq_cst, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.local.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(5) %e seq_cst, align 8

  ret void
}

; CHECK-LABEL: local_seq_cst_volatile
define void @local_seq_cst_volatile(ptr addrspace(5) %a, ptr addrspace(5) %b, ptr addrspace(5) %c, ptr addrspace(5) %d, ptr addrspace(5) %e) local_unnamed_addr {
  ; TODO: generate PTX that preserves Concurrent Forward Progress
  ;       by using PTX atomic operations.

  ; CHECK: ld.local.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(5) %a seq_cst, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.local.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(5) %a seq_cst, align 1

  ; CHECK: ld.local.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(5) %b seq_cst, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.local.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(5) %b seq_cst, align 2

  ; CHECK: ld.local.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(5) %c seq_cst, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.local.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(5) %c seq_cst, align 4

  ; CHECK: ld.local.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(5) %d seq_cst, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.local.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(5) %d seq_cst, align 8

  ; CHECK: ld.local.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(5) %e seq_cst, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.local.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(5) %e seq_cst, align 4

  ; CHECK: ld.local.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(5) %e seq_cst, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.local.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(5) %e seq_cst, align 8

  ; TODO: LLVM IR Verifier does not support atomics on vector types.

  ret void
}

; TODO: add plain,atomic,volatile,atomic volatile tests
;       for .const and .param statespaces