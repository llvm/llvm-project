; RUN: llc < %s -march=nvptx64 -mcpu=sm_70 -mattr=+ptx82 | FileCheck %s
; RUN: %if ptxas-12.2 %{ llc < %s -march=nvptx64 -mcpu=sm_70 -mattr=+ptx82 | %ptxas-verify -arch=sm_70 %}

; CHECK-LABEL: generic_plain
define void @generic_plain(ptr %a, ptr %b, ptr %c, ptr %d) local_unnamed_addr {
  ; CHECK: ld.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load i8, ptr %a
  %a.add = add i8 %a.load, 1
  ; CHECK: st.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store i8 %a.add, ptr %a

  ; CHECK: ld.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load i16, ptr %b
  %b.add = add i16 %b.load, 1
  ; CHECK: st.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store i16 %b.add, ptr %b

  ; CHECK: ld.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load i32, ptr %c
  %c.add = add i32 %c.load, 1
  ; CHECK: st.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store i32 %c.add, ptr %c

  ; CHECK: ld.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load i64, ptr %d
  %d.add = add i64 %d.load, 1
  ; CHECK: st.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store i64 %d.add, ptr %d

  ; CHECK: ld.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load float, ptr %c
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store float %e.add, ptr %c

  ; CHECK: ld.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load double, ptr %c
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store double %f.add, ptr %c

  ret void
}

; CHECK-LABEL: generic_volatile
define void @generic_volatile(ptr %a, ptr %b, ptr %c, ptr %d) local_unnamed_addr {
  ; CHECK: ld.volatile.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load volatile i8, ptr %a
  %a.add = add i8 %a.load, 1
  ; CHECK: st.volatile.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store volatile i8 %a.add, ptr %a

  ; CHECK: ld.volatile.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load volatile i16, ptr %b
  %b.add = add i16 %b.load, 1
  ; CHECK: st.volatile.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store volatile i16 %b.add, ptr %b

  ; CHECK: ld.volatile.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load volatile i32, ptr %c
  %c.add = add i32 %c.load, 1
  ; CHECK: st.volatile.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store volatile i32 %c.add, ptr %c

  ; CHECK: ld.volatile.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load volatile i64, ptr %d
  %d.add = add i64 %d.load, 1
  ; CHECK: st.volatile.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store volatile i64 %d.add, ptr %d

  ; CHECK: ld.volatile.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load volatile float, ptr %c
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.volatile.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store volatile float %e.add, ptr %c

  ; CHECK: ld.volatile.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load volatile double, ptr %c
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.volatile.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store volatile double %f.add, ptr %c

  ret void
}

; CHECK-LABEL: generic_unordered
define void @generic_unordered(ptr %a, ptr %b, ptr %c, ptr %d, ptr %e) local_unnamed_addr {
  ; CHECK: ld.relaxed.sys.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr %a unordered, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.relaxed.sys.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr %a unordered, align 1

  ; CHECK: ld.relaxed.sys.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr %b unordered, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.relaxed.sys.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr %b unordered, align 2

  ; CHECK: ld.relaxed.sys.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr %c unordered, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.relaxed.sys.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr %c unordered, align 4

  ; CHECK: ld.relaxed.sys.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr %d unordered, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.relaxed.sys.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr %d unordered, align 8

  ; CHECK: ld.relaxed.sys.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr %e unordered, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.relaxed.sys.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr %e unordered, align 4

  ; CHECK: ld.relaxed.sys.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr %e unordered, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.relaxed.sys.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr %e unordered, align 8

  ret void
}

; CHECK-LABEL: generic_monotonic
define void @generic_monotonic(ptr %a, ptr %b, ptr %c, ptr %d, ptr %e) local_unnamed_addr {
  ; CHECK: ld.relaxed.sys.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr %a monotonic, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.relaxed.sys.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr %a monotonic, align 1

  ; CHECK: ld.relaxed.sys.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr %b monotonic, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.relaxed.sys.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr %b monotonic, align 2

  ; CHECK: ld.relaxed.sys.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr %c monotonic, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.relaxed.sys.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr %c monotonic, align 4

  ; CHECK: ld.relaxed.sys.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr %d monotonic, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.relaxed.sys.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr %d monotonic, align 8

  ; CHECK: ld.relaxed.sys.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr %e monotonic, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.relaxed.sys.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr %e monotonic, align 4

  ; CHECK: ld.relaxed.sys.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr %e monotonic, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.relaxed.sys.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr %e monotonic, align 8

  ret void
}

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

; CHECK-LABEL: generic_unordered_volatile
define void @generic_unordered_volatile(ptr %a, ptr %b, ptr %c, ptr %d, ptr %e) local_unnamed_addr {
  ; CHECK: ld.volatile.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr %a unordered, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.volatile.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr %a unordered, align 1

  ; CHECK: ld.volatile.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr %b unordered, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.volatile.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr %b unordered, align 2

  ; CHECK: ld.volatile.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr %c unordered, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.volatile.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr %c unordered, align 4

  ; CHECK: ld.volatile.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr %d unordered, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.volatile.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr %d unordered, align 8

  ; CHECK: ld.volatile.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr %e unordered, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.volatile.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr %e unordered, align 4

  ; CHECK: ld.volatile.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr %e unordered, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.volatile.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr %e unordered, align 8

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

; CHECK-LABEL: generic_monotonic_volatile
define void @generic_monotonic_volatile(ptr %a, ptr %b, ptr %c, ptr %d, ptr %e) local_unnamed_addr {
  ; CHECK: ld.volatile.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr %a monotonic, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.volatile.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr %a monotonic, align 1

  ; CHECK: ld.volatile.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr %b monotonic, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.volatile.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr %b monotonic, align 2

  ; CHECK: ld.volatile.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr %c monotonic, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.volatile.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr %c monotonic, align 4

  ; CHECK: ld.volatile.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr %d monotonic, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.volatile.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr %d monotonic, align 8

  ; CHECK: ld.volatile.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr %e monotonic, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.volatile.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr %e monotonic, align 4

  ; CHECK: ld.volatile.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr %e monotonic, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.volatile.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr %e monotonic, align 8

  ret void
}

;; global statespace

; CHECK-LABEL: global_plain
define void @global_plain(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d) local_unnamed_addr {
  ; CHECK: ld.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load i8, ptr addrspace(1) %a
  %a.add = add i8 %a.load, 1
  ; CHECK: st.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store i8 %a.add, ptr addrspace(1) %a

  ; CHECK: ld.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load i16, ptr addrspace(1) %b
  %b.add = add i16 %b.load, 1
  ; CHECK: st.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store i16 %b.add, ptr addrspace(1) %b

  ; CHECK: ld.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load i32, ptr addrspace(1) %c
  %c.add = add i32 %c.load, 1
  ; CHECK: st.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store i32 %c.add, ptr addrspace(1) %c

  ; CHECK: ld.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load i64, ptr addrspace(1) %d
  %d.add = add i64 %d.load, 1
  ; CHECK: st.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store i64 %d.add, ptr addrspace(1) %d

  ; CHECK: ld.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load float, ptr addrspace(1) %c
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store float %e.add, ptr addrspace(1) %c

  ; CHECK: ld.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load double, ptr addrspace(1) %c
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store double %f.add, ptr addrspace(1) %c

  ret void
}

; CHECK-LABEL: global_volatile
define void @global_volatile(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d) local_unnamed_addr {
  ; CHECK: ld.volatile.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load volatile i8, ptr addrspace(1) %a
  %a.add = add i8 %a.load, 1
  ; CHECK: st.volatile.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store volatile i8 %a.add, ptr addrspace(1) %a

  ; CHECK: ld.volatile.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load volatile i16, ptr addrspace(1) %b
  %b.add = add i16 %b.load, 1
  ; CHECK: st.volatile.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store volatile i16 %b.add, ptr addrspace(1) %b

  ; CHECK: ld.volatile.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load volatile i32, ptr addrspace(1) %c
  %c.add = add i32 %c.load, 1
  ; CHECK: st.volatile.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store volatile i32 %c.add, ptr addrspace(1) %c

  ; CHECK: ld.volatile.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load volatile i64, ptr addrspace(1) %d
  %d.add = add i64 %d.load, 1
  ; CHECK: st.volatile.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store volatile i64 %d.add, ptr addrspace(1) %d

  ; CHECK: ld.volatile.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load volatile float, ptr addrspace(1) %c
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.volatile.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store volatile float %e.add, ptr addrspace(1) %c

  ; CHECK: ld.volatile.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load volatile double, ptr addrspace(1) %c
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.volatile.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store volatile double %f.add, ptr addrspace(1) %c

  ret void
}

; CHECK-LABEL: global_unordered
define void @global_unordered(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %e) local_unnamed_addr {
  ; CHECK: ld.relaxed.sys.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(1) %a unordered, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.relaxed.sys.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(1) %a unordered, align 1

  ; CHECK: ld.relaxed.sys.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(1) %b unordered, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.relaxed.sys.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(1) %b unordered, align 2

  ; CHECK: ld.relaxed.sys.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(1) %c unordered, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.relaxed.sys.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(1) %c unordered, align 4

  ; CHECK: ld.relaxed.sys.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(1) %d unordered, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.relaxed.sys.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(1) %d unordered, align 8

  ; CHECK: ld.relaxed.sys.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(1) %e unordered, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.relaxed.sys.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(1) %e unordered, align 4

  ; CHECK: ld.relaxed.sys.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(1) %e unordered, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.relaxed.sys.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(1) %e unordered, align 8

  ret void
}

; CHECK-LABEL: global_monotonic
define void @global_monotonic(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %e) local_unnamed_addr {
  ; CHECK: ld.relaxed.sys.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(1) %a monotonic, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.relaxed.sys.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(1) %a monotonic, align 1

  ; CHECK: ld.relaxed.sys.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(1) %b monotonic, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.relaxed.sys.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(1) %b monotonic, align 2

  ; CHECK: ld.relaxed.sys.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(1) %c monotonic, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.relaxed.sys.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(1) %c monotonic, align 4

  ; CHECK: ld.relaxed.sys.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(1) %d monotonic, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.relaxed.sys.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(1) %d monotonic, align 8

  ; CHECK: ld.relaxed.sys.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(1) %e monotonic, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.relaxed.sys.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(1) %e monotonic, align 4

  ; CHECK: ld.relaxed.sys.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(1) %e monotonic, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.relaxed.sys.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(1) %e monotonic, align 8

  ret void
}

; CHECK-LABEL: global_unordered_volatile
define void @global_unordered_volatile(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %e) local_unnamed_addr {
  ; CHECK: ld.mmio.relaxed.sys.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(1) %a unordered, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.mmio.relaxed.sys.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(1) %a unordered, align 1

  ; CHECK: ld.mmio.relaxed.sys.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(1) %b unordered, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.mmio.relaxed.sys.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(1) %b unordered, align 2

  ; CHECK: ld.mmio.relaxed.sys.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(1) %c unordered, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.mmio.relaxed.sys.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(1) %c unordered, align 4

  ; CHECK: ld.mmio.relaxed.sys.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(1) %d unordered, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.mmio.relaxed.sys.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(1) %d unordered, align 8

  ; CHECK: ld.mmio.relaxed.sys.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(1) %e unordered, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.mmio.relaxed.sys.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(1) %e unordered, align 4

  ; CHECK: ld.mmio.relaxed.sys.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(1) %e unordered, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.mmio.relaxed.sys.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(1) %e unordered, align 8

  ret void
}

; CHECK-LABEL: global_monotonic_volatile
define void @global_monotonic_volatile(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %e) local_unnamed_addr {
  ; CHECK: ld.mmio.relaxed.sys.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(1) %a monotonic, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.mmio.relaxed.sys.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(1) %a monotonic, align 1

  ; CHECK: ld.mmio.relaxed.sys.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(1) %b monotonic, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.mmio.relaxed.sys.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(1) %b monotonic, align 2

  ; CHECK: ld.mmio.relaxed.sys.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(1) %c monotonic, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.mmio.relaxed.sys.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(1) %c monotonic, align 4

  ; CHECK: ld.mmio.relaxed.sys.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(1) %d monotonic, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.mmio.relaxed.sys.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(1) %d monotonic, align 8

  ; CHECK: ld.mmio.relaxed.sys.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(1) %e monotonic, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.mmio.relaxed.sys.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(1) %e monotonic, align 4

  ; CHECK: ld.mmio.relaxed.sys.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(1) %e monotonic, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.mmio.relaxed.sys.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(1) %e monotonic, align 8

  ret void
}

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

;; shared statespace

; CHECK-LABEL: shared_plain
define void @shared_plain(ptr addrspace(3) %a, ptr addrspace(3) %b, ptr addrspace(3) %c, ptr addrspace(3) %d) local_unnamed_addr {
  ; CHECK: ld.shared.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load i8, ptr addrspace(3) %a
  %a.add = add i8 %a.load, 1
  ; CHECK: st.shared.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store i8 %a.add, ptr addrspace(3) %a

  ; CHECK: ld.shared.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load i16, ptr addrspace(3) %b
  %b.add = add i16 %b.load, 1
  ; CHECK: st.shared.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store i16 %b.add, ptr addrspace(3) %b

  ; CHECK: ld.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load i32, ptr addrspace(3) %c
  %c.add = add i32 %c.load, 1
  ; CHECK: st.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store i32 %c.add, ptr addrspace(3) %c

  ; CHECK: ld.shared.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load i64, ptr addrspace(3) %d
  %d.add = add i64 %d.load, 1
  ; CHECK: st.shared.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store i64 %d.add, ptr addrspace(3) %d

  ; CHECK: ld.shared.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load float, ptr addrspace(3) %c
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.shared.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store float %e.add, ptr addrspace(3) %c

  ; CHECK: ld.shared.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load double, ptr addrspace(3) %c
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.shared.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store double %f.add, ptr addrspace(3) %c

  ret void
}

; CHECK-LABEL: shared_volatile
define void @shared_volatile(ptr addrspace(3) %a, ptr addrspace(3) %b, ptr addrspace(3) %c, ptr addrspace(3) %d) local_unnamed_addr {
  ; CHECK: ld.volatile.shared.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load volatile i8, ptr addrspace(3) %a
  %a.add = add i8 %a.load, 1
  ; CHECK: st.volatile.shared.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store volatile i8 %a.add, ptr addrspace(3) %a

  ; CHECK: ld.volatile.shared.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load volatile i16, ptr addrspace(3) %b
  %b.add = add i16 %b.load, 1
  ; CHECK: st.volatile.shared.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store volatile i16 %b.add, ptr addrspace(3) %b

  ; CHECK: ld.volatile.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load volatile i32, ptr addrspace(3) %c
  %c.add = add i32 %c.load, 1
  ; CHECK: st.volatile.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store volatile i32 %c.add, ptr addrspace(3) %c

  ; CHECK: ld.volatile.shared.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load volatile i64, ptr addrspace(3) %d
  %d.add = add i64 %d.load, 1
  ; CHECK: st.volatile.shared.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store volatile i64 %d.add, ptr addrspace(3) %d

  ; CHECK: ld.volatile.shared.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load volatile float, ptr addrspace(3) %c
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.volatile.shared.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store volatile float %e.add, ptr addrspace(3) %c

  ; CHECK: ld.volatile.shared.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load volatile double, ptr addrspace(3) %c
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.volatile.shared.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store volatile double %f.add, ptr addrspace(3) %c

  ret void
}

; CHECK-LABEL: shared_unordered
define void @shared_unordered(ptr addrspace(3) %a, ptr addrspace(3) %b, ptr addrspace(3) %c, ptr addrspace(3) %d, ptr addrspace(3) %e) local_unnamed_addr {
  ; CHECK: ld.relaxed.sys.shared.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(3) %a unordered, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.relaxed.sys.shared.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(3) %a unordered, align 1

  ; CHECK: ld.relaxed.sys.shared.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(3) %b unordered, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.relaxed.sys.shared.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(3) %b unordered, align 2

  ; CHECK: ld.relaxed.sys.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(3) %c unordered, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.relaxed.sys.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(3) %c unordered, align 4

  ; CHECK: ld.relaxed.sys.shared.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(3) %d unordered, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.relaxed.sys.shared.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(3) %d unordered, align 8

  ; CHECK: ld.relaxed.sys.shared.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(3) %e unordered, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.relaxed.sys.shared.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(3) %e unordered, align 4

  ; CHECK: ld.relaxed.sys.shared.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(3) %e unordered, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.relaxed.sys.shared.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(3) %e unordered, align 8

  ret void
}

; CHECK-LABEL: shared_unordered_volatile
define void @shared_unordered_volatile(ptr addrspace(3) %a, ptr addrspace(3) %b, ptr addrspace(3) %c, ptr addrspace(3) %d, ptr addrspace(3) %e) local_unnamed_addr {
  ; CHECK: ld.volatile.shared.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(3) %a unordered, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.volatile.shared.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(3) %a unordered, align 1

  ; CHECK: ld.volatile.shared.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(3) %b unordered, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.volatile.shared.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(3) %b unordered, align 2

  ; CHECK: ld.volatile.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(3) %c unordered, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.volatile.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(3) %c unordered, align 4

  ; CHECK: ld.volatile.shared.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(3) %d unordered, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.volatile.shared.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(3) %d unordered, align 8

  ; CHECK: ld.volatile.shared.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(3) %e unordered, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.volatile.shared.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(3) %e unordered, align 4

  ; CHECK: ld.volatile.shared.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(3) %e unordered, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.volatile.shared.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(3) %e unordered, align 8

  ret void
}

; CHECK-LABEL: shared_monotonic
define void @shared_monotonic(ptr addrspace(3) %a, ptr addrspace(3) %b, ptr addrspace(3) %c, ptr addrspace(3) %d, ptr addrspace(3) %e) local_unnamed_addr {
  ; CHECK: ld.relaxed.sys.shared.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(3) %a monotonic, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.relaxed.sys.shared.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(3) %a monotonic, align 1

  ; CHECK: ld.relaxed.sys.shared.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(3) %b monotonic, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.relaxed.sys.shared.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(3) %b monotonic, align 2

  ; CHECK: ld.relaxed.sys.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(3) %c monotonic, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.relaxed.sys.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(3) %c monotonic, align 4

  ; CHECK: ld.relaxed.sys.shared.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(3) %d monotonic, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.relaxed.sys.shared.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(3) %d monotonic, align 8

  ; CHECK: ld.relaxed.sys.shared.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(3) %e monotonic, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.relaxed.sys.shared.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(3) %e monotonic, align 4

  ; CHECK: ld.relaxed.sys.shared.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(3) %e monotonic, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.relaxed.sys.shared.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(3) %e monotonic, align 8

  ret void
}

; CHECK-LABEL: shared_monotonic_volatile
define void @shared_monotonic_volatile(ptr addrspace(3) %a, ptr addrspace(3) %b, ptr addrspace(3) %c, ptr addrspace(3) %d, ptr addrspace(3) %e) local_unnamed_addr {
  ; CHECK: ld.volatile.shared.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(3) %a monotonic, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.volatile.shared.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(3) %a monotonic, align 1

  ; CHECK: ld.volatile.shared.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(3) %b monotonic, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.volatile.shared.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(3) %b monotonic, align 2

  ; CHECK: ld.volatile.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(3) %c monotonic, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.volatile.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(3) %c monotonic, align 4

  ; CHECK: ld.volatile.shared.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(3) %d monotonic, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.volatile.shared.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(3) %d monotonic, align 8

  ; CHECK: ld.volatile.shared.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(3) %e monotonic, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.volatile.shared.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(3) %e monotonic, align 4

  ; CHECK: ld.volatile.shared.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(3) %e monotonic, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.volatile.shared.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(3) %e monotonic, align 8

  ret void
}

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

;; local statespace

; CHECK-LABEL: local_plain
define void @local_plain(ptr addrspace(5) %a, ptr addrspace(5) %b, ptr addrspace(5) %c, ptr addrspace(5) %d) local_unnamed_addr {
  ; CHECK: ld.local.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load i8, ptr addrspace(5) %a
  %a.add = add i8 %a.load, 1
  ; CHECK: st.local.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store i8 %a.add, ptr addrspace(5) %a

  ; CHECK: ld.local.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load i16, ptr addrspace(5) %b
  %b.add = add i16 %b.load, 1
  ; CHECK: st.local.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store i16 %b.add, ptr addrspace(5) %b

  ; CHECK: ld.local.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load i32, ptr addrspace(5) %c
  %c.add = add i32 %c.load, 1
  ; CHECK: st.local.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store i32 %c.add, ptr addrspace(5) %c

  ; CHECK: ld.local.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load i64, ptr addrspace(5) %d
  %d.add = add i64 %d.load, 1
  ; CHECK: st.local.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store i64 %d.add, ptr addrspace(5) %d

  ; CHECK: ld.local.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load float, ptr addrspace(5) %c
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.local.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store float %e.add, ptr addrspace(5) %c

  ; CHECK: ld.local.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load double, ptr addrspace(5) %c
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.local.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store double %f.add, ptr addrspace(5) %c

  ret void
}

; CHECK-LABEL: local_volatile
define void @local_volatile(ptr addrspace(5) %a, ptr addrspace(5) %b, ptr addrspace(5) %c, ptr addrspace(5) %d) local_unnamed_addr {
  ; CHECK: ld.local.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load volatile i8, ptr addrspace(5) %a
  %a.add = add i8 %a.load, 1
  ; CHECK: st.local.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store volatile i8 %a.add, ptr addrspace(5) %a

  ; CHECK: ld.local.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load volatile i16, ptr addrspace(5) %b
  %b.add = add i16 %b.load, 1
  ; CHECK: st.local.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store volatile i16 %b.add, ptr addrspace(5) %b

  ; CHECK: ld.local.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load volatile i32, ptr addrspace(5) %c
  %c.add = add i32 %c.load, 1
  ; CHECK: st.local.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store volatile i32 %c.add, ptr addrspace(5) %c

  ; CHECK: ld.local.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load volatile i64, ptr addrspace(5) %d
  %d.add = add i64 %d.load, 1
  ; CHECK: st.local.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store volatile i64 %d.add, ptr addrspace(5) %d

  ; CHECK: ld.local.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load volatile float, ptr addrspace(5) %c
  %e.add = fadd float %e.load, 1.
  ; CHECK: st.local.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store volatile float %e.add, ptr addrspace(5) %c

  ; CHECK: ld.local.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load volatile double, ptr addrspace(5) %c
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.local.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store volatile double %f.add, ptr addrspace(5) %c

  ret void
}

; CHECK-LABEL: local_unordered
define void @local_unordered(ptr addrspace(5) %a, ptr addrspace(5) %b, ptr addrspace(5) %c, ptr addrspace(5) %d, ptr addrspace(5) %e) local_unnamed_addr {
  ; CHECK: ld.local.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(5) %a unordered, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.local.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(5) %a unordered, align 1

  ; CHECK: ld.local.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(5) %b unordered, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.local.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(5) %b unordered, align 2

  ; CHECK: ld.local.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(5) %c unordered, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.local.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(5) %c unordered, align 4

  ; CHECK: ld.local.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(5) %d unordered, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.local.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(5) %d unordered, align 8

  ; CHECK: ld.local.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(5) %e unordered, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.local.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(5) %e unordered, align 4

  ; CHECK: ld.local.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(5) %e unordered, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.local.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(5) %e unordered, align 8

  ret void
}

; CHECK-LABEL: local_unordered_volatile
define void @local_unordered_volatile(ptr addrspace(5) %a, ptr addrspace(5) %b, ptr addrspace(5) %c, ptr addrspace(5) %d, ptr addrspace(5) %e) local_unnamed_addr {
  ; CHECK: ld.local.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(5) %a unordered, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.local.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(5) %a unordered, align 1

  ; CHECK: ld.local.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(5) %b unordered, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.local.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(5) %b unordered, align 2

  ; CHECK: ld.local.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(5) %c unordered, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.local.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(5) %c unordered, align 4

  ; CHECK: ld.local.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(5) %d unordered, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.local.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(5) %d unordered, align 8

  ; CHECK: ld.local.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(5) %e unordered, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.local.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(5) %e unordered, align 4

  ; CHECK: ld.local.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(5) %e unordered, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.local.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(5) %e unordered, align 8

  ret void
}

; CHECK-LABEL: local_monotonic
define void @local_monotonic(ptr addrspace(5) %a, ptr addrspace(5) %b, ptr addrspace(5) %c, ptr addrspace(5) %d, ptr addrspace(5) %e) local_unnamed_addr {
  ; CHECK: ld.local.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(5) %a monotonic, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.local.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(5) %a monotonic, align 1

  ; CHECK: ld.local.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(5) %b monotonic, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.local.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(5) %b monotonic, align 2

  ; CHECK: ld.local.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(5) %c monotonic, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.local.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(5) %c monotonic, align 4

  ; CHECK: ld.local.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(5) %d monotonic, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.local.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(5) %d monotonic, align 8

  ; CHECK: ld.local.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(5) %e monotonic, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.local.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(5) %e monotonic, align 4

  ; CHECK: ld.local.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(5) %e monotonic, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.local.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(5) %e monotonic, align 8

  ret void
}

; CHECK-LABEL: local_monotonic_volatile
define void @local_monotonic_volatile(ptr addrspace(5) %a, ptr addrspace(5) %b, ptr addrspace(5) %c, ptr addrspace(5) %d, ptr addrspace(5) %e) local_unnamed_addr {
  ; CHECK: ld.local.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(5) %a monotonic, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.local.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(5) %a monotonic, align 1

  ; CHECK: ld.local.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(5) %b monotonic, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.local.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(5) %b monotonic, align 2

  ; CHECK: ld.local.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(5) %c monotonic, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.local.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(5) %c monotonic, align 4

  ; CHECK: ld.local.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(5) %d monotonic, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.local.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(5) %d monotonic, align 8

  ; CHECK: ld.local.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(5) %e monotonic, align 4
  %e.add = fadd float %e.load, 1.0
  ; CHECK: st.local.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(5) %e monotonic, align 4

  ; CHECK: ld.local.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(5) %e monotonic, align 8
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.local.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(5) %e monotonic, align 8

  ret void
}

; CHECK-LABEL: local_acq_rel
define void @local_acq_rel(ptr addrspace(5) %a, ptr addrspace(5) %b, ptr addrspace(5) %c, ptr addrspace(5) %d, ptr addrspace(5) %e) local_unnamed_addr {
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
