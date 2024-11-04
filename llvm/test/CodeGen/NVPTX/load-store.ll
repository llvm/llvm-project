; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck -check-prefixes=CHECK,SM60 %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}
; RUN: llc < %s -march=nvptx64 -mcpu=sm_70 -mattr=+ptx82 | FileCheck %s -check-prefixes=CHECK,SM70
; RUN: %if ptxas-12.2 %{ llc < %s -march=nvptx64 -mcpu=sm_70 -mattr=+ptx82 | %ptxas-verify -arch=sm_70 %}

; TODO: add i1, <8 x i8>, and <6 x i8> vector tests.

; TODO: add test for vectors that exceed 128-bit length
; Per https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#vectors
; vectors cannot exceed 128-bit in length, i.e., .v4.u64 is not allowed.

; generic statespace

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
  %f.load = load double, ptr %d
  %f.add = fadd double %f.load, 1.
  ; CHECK: st.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store double %f.add, ptr %d

  ; TODO: make the lowering of this weak vector ops consistent with
  ;       the ones of the next tests. This test lowers to a weak PTX
  ;       vector op, but next test lowers to a vector PTX op.
  ; CHECK: ld.v2.u8 {%rs{{[0-9]+}}, %rs{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %h.load = load <2 x i8>, ptr %b
  %h.add = add <2 x i8> %h.load, <i8 1, i8 1>
  ; CHECK: st.v2.u8 [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}}
  store <2 x i8> %h.add, ptr %b

  ; TODO: make the lowering of this weak vector ops consistent with
  ;       the ones of the previous test. This test lowers to a weak
  ;       PTX scalar op, but prior test lowers to a vector PTX op.
  ; CHECK: ld.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %i.load = load <4 x i8>, ptr %c
  %i.add = add <4 x i8> %i.load, <i8 1, i8 1, i8 1, i8 1>
  ; CHECK: st.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store <4 x i8> %i.add, ptr %c

  ; CHECK: ld.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %j.load = load <2 x i16>, ptr %c
  %j.add = add <2 x i16> %j.load, <i16 1, i16 1>
  ; CHECK: st.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store <2 x i16> %j.add, ptr %c

  ; CHECK: ld.v4.u16 {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %k.load = load <4 x i16>, ptr %d
  %k.add = add <4 x i16> %k.load, <i16 1, i16 1, i16 1, i16 1>
  ; CHECK: st.v4.u16 [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}}
  store <4 x i16> %k.add, ptr %d

  ; CHECK: ld.v2.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %l.load = load <2 x i32>, ptr %d
  %l.add = add <2 x i32> %l.load, <i32 1, i32 1>
  ; CHECK: st.v2.u32 [%rd{{[0-9]+}}], {%r{{[0-9]+}}, %r{{[0-9]+}}}
  store <2 x i32> %l.add, ptr %d

  ; CHECK: ld.v4.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %m.load = load <4 x i32>, ptr %d
  %m.add = add <4 x i32> %m.load, <i32 1, i32 1, i32 1, i32 1>
  ; CHECK: st.v4.u32 [%rd{{[0-9]+}}], {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}
  store <4 x i32> %m.add, ptr %d

  ; CHECK: ld.v2.u64 {%rd{{[0-9]+}}, %rd{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %n.load = load <2 x i64>, ptr %d
  %n.add = add <2 x i64> %n.load, <i64 1, i64 1>
  ; CHECK: st.v2.u64 [%rd{{[0-9]+}}], {%rd{{[0-9]+}}, %rd{{[0-9]+}}}
  store <2 x i64> %n.add, ptr %d

  ; CHECK: ld.v2.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %o.load = load <2 x float>, ptr %d
  %o.add = fadd <2 x float> %o.load, <float 1., float 1.>
  ; CHECK: st.v2.f32 [%rd{{[0-9]+}}], {%f{{[0-9]+}}, %f{{[0-9]+}}}
  store <2 x float> %o.add, ptr %d

  ; CHECK: ld.v4.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %p.load = load <4 x float>, ptr %d
  %p.add = fadd <4 x float> %p.load, <float 1., float 1., float 1., float 1.>
  ; CHECK: st.v4.f32 [%rd{{[0-9]+}}], {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}
  store <4 x float> %p.add, ptr %d

  ; CHECK: ld.v2.f64 {%fd{{[0-9]+}}, %fd{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %q.load = load <2 x double>, ptr %d
  %q.add = fadd <2 x double> %q.load, <double 1., double 1.>
  ; CHECK: st.v2.f64 [%rd{{[0-9]+}}], {%fd{{[0-9]+}}, %fd{{[0-9]+}}}
  store <2 x double> %q.add, ptr %d

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

  ; TODO: volatile, atomic, and volatile atomic memory operations on vector types.
  ; Currently, LLVM:
  ; - does not allow atomic operations on vectors.
  ; - it allows volatile operations but not clear what that means.
  ; Following both semantics make sense in general and PTX supports both:
  ; - volatile/atomic/volatile atomic applies to the whole vector
  ; - volatile/atomic/volatile atomic applies elementwise
  ; Actions required:
  ; - clarify LLVM semantics for volatile on vectors and align the NVPTX backend with those
  ;   Below tests show that the current implementation picks the semantics in an inconsistent way
  ;   * volatile <2 x i8> lowers to "elementwise volatile"
  ;   * <4 x i8> lowers to "full vector volatile"
  ; - provide support for vector atomics, e.g., by extending LLVM IR or via intrinsics
  ; - update tests in load-store-sm70.ll as well.

  ; TODO: make this operation consistent with the one for <4 x i8>
  ; This operation lowers to a "element wise volatile PTX operation".
  ; CHECK: ld.volatile.v2.u8 {%rs{{[0-9]+}}, %rs{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %h.load = load volatile <2 x i8>, ptr %b
  %h.add = add <2 x i8> %h.load, <i8 1, i8 1>
  ; CHECK: st.volatile.v2.u8 [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}}
  store volatile <2 x i8> %h.add, ptr %b

  ; TODO: make this operation consistent with the one for <2 x i8>
  ; This operation lowers to a "full vector volatile PTX operation".
  ; CHECK: ld.volatile.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %i.load = load volatile <4 x i8>, ptr %c
  %i.add = add <4 x i8> %i.load, <i8 1, i8 1, i8 1, i8 1>
  ; CHECK: st.volatile.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store volatile <4 x i8> %i.add, ptr %c

  ; CHECK: ld.volatile.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %j.load = load volatile <2 x i16>, ptr %c
  %j.add = add <2 x i16> %j.load, <i16 1, i16 1>
  ; CHECK: st.volatile.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store volatile <2 x i16> %j.add, ptr %c

  ; CHECK: ld.volatile.v4.u16 {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %k.load = load volatile <4 x i16>, ptr %d
  %k.add = add <4 x i16> %k.load, <i16 1, i16 1, i16 1, i16 1>
  ; CHECK: st.volatile.v4.u16 [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}}
  store volatile <4 x i16> %k.add, ptr %d

  ; CHECK: ld.volatile.v2.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %l.load = load volatile <2 x i32>, ptr %d
  %l.add = add <2 x i32> %l.load, <i32 1, i32 1>
  ; CHECK: st.volatile.v2.u32 [%rd{{[0-9]+}}], {%r{{[0-9]+}}, %r{{[0-9]+}}}
  store volatile <2 x i32> %l.add, ptr %d

  ; CHECK: ld.volatile.v4.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %m.load = load volatile <4 x i32>, ptr %d
  %m.add = add <4 x i32> %m.load, <i32 1, i32 1, i32 1, i32 1>
  ; CHECK: st.volatile.v4.u32 [%rd{{[0-9]+}}], {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}
  store volatile <4 x i32> %m.add, ptr %d

  ; CHECK: ld.volatile.v2.u64 {%rd{{[0-9]+}}, %rd{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %n.load = load volatile <2 x i64>, ptr %d
  %n.add = add <2 x i64> %n.load, <i64 1, i64 1>
  ; CHECK: st.volatile.v2.u64 [%rd{{[0-9]+}}], {%rd{{[0-9]+}}, %rd{{[0-9]+}}}
  store volatile <2 x i64> %n.add, ptr %d

  ; CHECK: ld.volatile.v2.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %o.load = load volatile <2 x float>, ptr %d
  %o.add = fadd <2 x float> %o.load, <float 1., float 1.>
  ; CHECK: st.volatile.v2.f32 [%rd{{[0-9]+}}], {%f{{[0-9]+}}, %f{{[0-9]+}}}
  store volatile <2 x float> %o.add, ptr %d

  ; CHECK: ld.volatile.v4.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %p.load = load volatile <4 x float>, ptr %d
  %p.add = fadd <4 x float> %p.load, <float 1., float 1., float 1., float 1.>
  ; CHECK: st.volatile.v4.f32 [%rd{{[0-9]+}}], {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}
  store volatile <4 x float> %p.add, ptr %d

  ; CHECK: ld.volatile.v2.f64 {%fd{{[0-9]+}}, %fd{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %q.load = load volatile <2 x double>, ptr %d
  %q.add = fadd <2 x double> %q.load, <double 1., double 1.>
  ; CHECK: st.volatile.v2.f64 [%rd{{[0-9]+}}], {%fd{{[0-9]+}}, %fd{{[0-9]+}}}
  store volatile <2 x double> %q.add, ptr %d

  ret void
}

; CHECK-LABEL: generic_monotonic
define void @generic_monotonic(ptr %a, ptr %b, ptr %c, ptr %d, ptr %e) local_unnamed_addr {
  ; SM60: ld.volatile.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr %a monotonic, align 1
  %a.add = add i8 %a.load, 1
  ; SM60: st.volatile.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  ; SM70: st.relaxed.sys.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr %a monotonic, align 1

  ; SM60: ld.volatile.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr %b monotonic, align 2
  %b.add = add i16 %b.load, 1
  ; SM60: st.volatile.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  ; SM70: st.relaxed.sys.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr %b monotonic, align 2

  ; SM60: ld.volatile.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr %c monotonic, align 4
  %c.add = add i32 %c.load, 1
  ; SM60: st.volatile.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  ; SM70: st.relaxed.sys.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr %c monotonic, align 4

  ; SM60: ld.volatile.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr %d monotonic, align 8
  %d.add = add i64 %d.load, 1
  ; SM60: st.volatile.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  ; SM70: st.relaxed.sys.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr %d monotonic, align 8

  ; SM60: ld.volatile.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr %e monotonic, align 4
  %e.add = fadd float %e.load, 1.0
  ; SM60: st.volatile.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  ; SM70: st.relaxed.sys.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr %e monotonic, align 4

  ; SM60: ld.volatile.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr %e monotonic, align 8
  %f.add = fadd double %f.load, 1.
  ; SM60: st.volatile.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  ; SM70: st.relaxed.sys.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr %e monotonic, align 8

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

; CHECK-LABEL: generic_unordered
define void @generic_unordered(ptr %a, ptr %b, ptr %c, ptr %d, ptr %e) local_unnamed_addr {
  ; SM60: ld.volatile.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr %a unordered, align 1
  %a.add = add i8 %a.load, 1
  ; SM60: st.volatile.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  ; SM70: st.relaxed.sys.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr %a unordered, align 1

  ; SM60: ld.volatile.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr %b unordered, align 2
  %b.add = add i16 %b.load, 1
  ; SM60: st.volatile.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  ; SM70: st.relaxed.sys.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr %b unordered, align 2

  ; SM60: ld.volatile.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr %c unordered, align 4
  %c.add = add i32 %c.load, 1
  ; SM60: st.volatile.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  ; SM70: st.relaxed.sys.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr %c unordered, align 4

  ; SM60: ld.volatile.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr %d unordered, align 8
  %d.add = add i64 %d.load, 1
  ; SM60: st.volatile.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  ; SM70: st.relaxed.sys.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr %d unordered, align 8

  ; SM60: ld.volatile.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr %e unordered, align 4
  %e.add = fadd float %e.load, 1.0
  ; SM60: st.volatile.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  ; SM70: st.relaxed.sys.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr %e unordered, align 4

  ; SM60: ld.volatile.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr %e unordered, align 8
  %f.add = fadd double %f.load, 1.
  ; SM60: st.volatile.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  ; SM70: st.relaxed.sys.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr %e unordered, align 8

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

  ; CHECK: ld.global.v2.u8 {%rs{{[0-9]+}}, %rs{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %h.load = load <2 x i8>, ptr addrspace(1) %b
  %h.add = add <2 x i8> %h.load, <i8 1, i8 1>
  ; CHECK: st.global.v2.u8 [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}}
  store <2 x i8> %h.add, ptr addrspace(1) %b

  ; CHECK: ld.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %i.load = load <4 x i8>, ptr addrspace(1) %c
  %i.add = add <4 x i8> %i.load, <i8 1, i8 1, i8 1, i8 1>
  ; CHECK: st.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store <4 x i8> %i.add, ptr addrspace(1) %c

  ; CHECK: ld.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %j.load = load <2 x i16>, ptr addrspace(1) %c
  %j.add = add <2 x i16> %j.load, <i16 1, i16 1>
  ; CHECK: st.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store <2 x i16> %j.add, ptr addrspace(1) %c

  ; CHECK: ld.global.v4.u16 {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %k.load = load <4 x i16>, ptr addrspace(1) %d
  %k.add = add <4 x i16> %k.load, <i16 1, i16 1, i16 1, i16 1>
  ; CHECK: st.global.v4.u16 [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}}
  store <4 x i16> %k.add, ptr addrspace(1) %d

  ; CHECK: ld.global.v2.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %l.load = load <2 x i32>, ptr addrspace(1) %d
  %l.add = add <2 x i32> %l.load, <i32 1, i32 1>
  ; CHECK: st.global.v2.u32 [%rd{{[0-9]+}}], {%r{{[0-9]+}}, %r{{[0-9]+}}}
  store <2 x i32> %l.add, ptr addrspace(1) %d

  ; CHECK: ld.global.v4.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %m.load = load <4 x i32>, ptr addrspace(1) %d
  %m.add = add <4 x i32> %m.load, <i32 1, i32 1, i32 1, i32 1>
  ; CHECK: st.global.v4.u32 [%rd{{[0-9]+}}], {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}
  store <4 x i32> %m.add, ptr addrspace(1) %d

  ; CHECK: ld.global.v2.u64 {%rd{{[0-9]+}}, %rd{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %n.load = load <2 x i64>, ptr addrspace(1) %d
  %n.add = add <2 x i64> %n.load, <i64 1, i64 1>
  ; CHECK: st.global.v2.u64 [%rd{{[0-9]+}}], {%rd{{[0-9]+}}, %rd{{[0-9]+}}}
  store <2 x i64> %n.add, ptr addrspace(1) %d

  ; CHECK: ld.global.v2.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %o.load = load <2 x float>, ptr addrspace(1) %d
  %o.add = fadd <2 x float> %o.load, <float 1., float 1.>
  ; CHECK: st.global.v2.f32 [%rd{{[0-9]+}}], {%f{{[0-9]+}}, %f{{[0-9]+}}}
  store <2 x float> %o.add, ptr addrspace(1) %d

  ; CHECK: ld.global.v4.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %p.load = load <4 x float>, ptr addrspace(1) %d
  %p.add = fadd <4 x float> %p.load, <float 1., float 1., float 1., float 1.>
  ; CHECK: st.global.v4.f32 [%rd{{[0-9]+}}], {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}
  store <4 x float> %p.add, ptr addrspace(1) %d

  ; CHECK: ld.global.v2.f64 {%fd{{[0-9]+}}, %fd{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %q.load = load <2 x double>, ptr addrspace(1) %d
  %q.add = fadd <2 x double> %q.load, <double 1., double 1.>
  ; CHECK: st.global.v2.f64 [%rd{{[0-9]+}}], {%fd{{[0-9]+}}, %fd{{[0-9]+}}}
  store <2 x double> %q.add, ptr addrspace(1) %d

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

  ; CHECK: ld.volatile.global.v2.u8 {%rs{{[0-9]+}}, %rs{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %h.load = load volatile <2 x i8>, ptr addrspace(1) %b
  %h.add = add <2 x i8> %h.load, <i8 1, i8 1>
  ; CHECK: st.volatile.global.v2.u8 [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}}
  store volatile<2 x i8> %h.add, ptr addrspace(1) %b

  ; CHECK: ld.volatile.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %i.load = load volatile <4 x i8>, ptr addrspace(1) %c
  %i.add = add <4 x i8> %i.load, <i8 1, i8 1, i8 1, i8 1>
  ; CHECK: st.volatile.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store volatile<4 x i8> %i.add, ptr addrspace(1) %c

  ; CHECK: ld.volatile.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %j.load = load volatile <2 x i16>, ptr addrspace(1) %c
  %j.add = add <2 x i16> %j.load, <i16 1, i16 1>
  ; CHECK: st.volatile.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store volatile<2 x i16> %j.add, ptr addrspace(1) %c

  ; CHECK: ld.volatile.global.v4.u16 {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %k.load = load volatile <4 x i16>, ptr addrspace(1) %d
  %k.add = add <4 x i16> %k.load, <i16 1, i16 1, i16 1, i16 1>
  ; CHECK: st.volatile.global.v4.u16 [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}}
  store volatile<4 x i16> %k.add, ptr addrspace(1) %d

  ; CHECK: ld.volatile.global.v2.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %l.load = load volatile <2 x i32>, ptr addrspace(1) %d
  %l.add = add <2 x i32> %l.load, <i32 1, i32 1>
  ; CHECK: st.volatile.global.v2.u32 [%rd{{[0-9]+}}], {%r{{[0-9]+}}, %r{{[0-9]+}}}
  store volatile<2 x i32> %l.add, ptr addrspace(1) %d

  ; CHECK: ld.volatile.global.v4.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %m.load = load volatile <4 x i32>, ptr addrspace(1) %d
  %m.add = add <4 x i32> %m.load, <i32 1, i32 1, i32 1, i32 1>
  ; CHECK: st.volatile.global.v4.u32 [%rd{{[0-9]+}}], {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}
  store volatile<4 x i32> %m.add, ptr addrspace(1) %d

  ; CHECK: ld.volatile.global.v2.u64 {%rd{{[0-9]+}}, %rd{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %n.load = load volatile <2 x i64>, ptr addrspace(1) %d
  %n.add = add <2 x i64> %n.load, <i64 1, i64 1>
  ; CHECK: st.volatile.global.v2.u64 [%rd{{[0-9]+}}], {%rd{{[0-9]+}}, %rd{{[0-9]+}}}
  store volatile<2 x i64> %n.add, ptr addrspace(1) %d

  ; CHECK: ld.volatile.global.v2.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %o.load = load volatile <2 x float>, ptr addrspace(1) %d
  %o.add = fadd <2 x float> %o.load, <float 1., float 1.>
  ; CHECK: st.volatile.global.v2.f32 [%rd{{[0-9]+}}], {%f{{[0-9]+}}, %f{{[0-9]+}}}
  store volatile<2 x float> %o.add, ptr addrspace(1) %d

  ; CHECK: ld.volatile.global.v4.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %p.load = load volatile <4 x float>, ptr addrspace(1) %d
  %p.add = fadd <4 x float> %p.load, <float 1., float 1., float 1., float 1.>
  ; CHECK: st.volatile.global.v4.f32 [%rd{{[0-9]+}}], {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}
  store volatile<4 x float> %p.add, ptr addrspace(1) %d

  ; CHECK: ld.volatile.global.v2.f64 {%fd{{[0-9]+}}, %fd{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %q.load = load volatile <2 x double>, ptr addrspace(1) %d
  %q.add = fadd <2 x double> %q.load, <double 1., double 1.>
  ; CHECK: st.volatile.global.v2.f64 [%rd{{[0-9]+}}], {%fd{{[0-9]+}}, %fd{{[0-9]+}}}
  store volatile<2 x double> %q.add, ptr addrspace(1) %d

  ret void
}

; CHECK-LABEL: global_monotonic
define void @global_monotonic(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %e) local_unnamed_addr {
  ; SM60: ld.volatile.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(1) %a monotonic, align 1
  %a.add = add i8 %a.load, 1
  ; SM60: st.volatile.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  ; SM70: st.relaxed.sys.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(1) %a monotonic, align 1

  ; SM60: ld.volatile.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(1) %b monotonic, align 2
  %b.add = add i16 %b.load, 1
  ; SM60: st.volatile.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  ; SM70: st.relaxed.sys.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(1) %b monotonic, align 2

  ; SM60: ld.volatile.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(1) %c monotonic, align 4
  %c.add = add i32 %c.load, 1
  ; SM60: st.volatile.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  ; SM70: st.relaxed.sys.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(1) %c monotonic, align 4

  ; SM60: ld.volatile.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(1) %d monotonic, align 8
  %d.add = add i64 %d.load, 1
  ; SM60: st.volatile.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  ; SM70: st.relaxed.sys.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(1) %d monotonic, align 8

  ; SM60: ld.volatile.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(1) %e monotonic, align 4
  %e.add = fadd float %e.load, 1.0
  ; SM60: st.volatile.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  ; SM70: st.relaxed.sys.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(1) %e monotonic, align 4

  ; SM60: ld.volatile.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(1) %e monotonic, align 8
  %f.add = fadd double %f.load, 1.
  ; SM60: st.volatile.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  ; SM70: st.relaxed.sys.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(1) %e monotonic, align 8

  ret void
}

; CHECK-LABEL: global_monotonic_volatile
define void @global_monotonic_volatile(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %e) local_unnamed_addr {
  ; SM60: ld.volatile.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.mmio.relaxed.sys.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(1) %a monotonic, align 1
  %a.add = add i8 %a.load, 1
  ; SM60: st.volatile.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  ; SM70: st.mmio.relaxed.sys.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(1) %a monotonic, align 1

  ; SM60: ld.volatile.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.mmio.relaxed.sys.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(1) %b monotonic, align 2
  %b.add = add i16 %b.load, 1
  ; SM60: st.volatile.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  ; SM70: st.mmio.relaxed.sys.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(1) %b monotonic, align 2

  ; SM60: ld.volatile.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.mmio.relaxed.sys.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(1) %c monotonic, align 4
  %c.add = add i32 %c.load, 1
  ; SM60: st.volatile.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  ; SM70: st.mmio.relaxed.sys.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(1) %c monotonic, align 4

  ; SM60: ld.volatile.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.mmio.relaxed.sys.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(1) %d monotonic, align 8
  %d.add = add i64 %d.load, 1
  ; SM60: st.volatile.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  ; SM70: st.mmio.relaxed.sys.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(1) %d monotonic, align 8

  ; SM60: ld.volatile.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.mmio.relaxed.sys.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(1) %e monotonic, align 4
  %e.add = fadd float %e.load, 1.0
  ; SM60: st.volatile.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  ; SM70: st.mmio.relaxed.sys.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(1) %e monotonic, align 4

  ; SM60: ld.volatile.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.mmio.relaxed.sys.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(1) %e monotonic, align 8
  %f.add = fadd double %f.load, 1.
  ; SM60: st.volatile.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  ; SM70: st.mmio.relaxed.sys.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(1) %e monotonic, align 8

  ret void
}

; CHECK-LABEL: global_unordered
define void @global_unordered(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %e) local_unnamed_addr {
  ; SM60: ld.volatile.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(1) %a unordered, align 1
  %a.add = add i8 %a.load, 1
  ; SM60: st.volatile.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  ; SM70: st.relaxed.sys.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(1) %a unordered, align 1

  ; SM60: ld.volatile.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(1) %b unordered, align 2
  %b.add = add i16 %b.load, 1
  ; SM60: st.volatile.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  ; SM70: st.relaxed.sys.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(1) %b unordered, align 2

  ; SM60: ld.volatile.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(1) %c unordered, align 4
  %c.add = add i32 %c.load, 1
  ; SM60: st.volatile.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  ; SM70: st.relaxed.sys.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(1) %c unordered, align 4

  ; SM60: ld.volatile.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(1) %d unordered, align 8
  %d.add = add i64 %d.load, 1
  ; SM60: st.volatile.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  ; SM70: st.relaxed.sys.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(1) %d unordered, align 8

  ; SM60: ld.volatile.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(1) %e unordered, align 4
  %e.add = fadd float %e.load, 1.0
  ; SM60: st.volatile.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  ; SM70: st.relaxed.sys.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(1) %e unordered, align 4

  ; SM60: ld.volatile.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(1) %e unordered, align 8
  %f.add = fadd double %f.load, 1.
  ; SM60: st.volatile.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  ; SM70: st.relaxed.sys.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic double %f.add, ptr addrspace(1) %e unordered, align 8

  ret void
}

; CHECK-LABEL: global_unordered_volatile
define void @global_unordered_volatile(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %e) local_unnamed_addr {
  ; SM60: ld.volatile.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.mmio.relaxed.sys.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic volatile i8, ptr addrspace(1) %a unordered, align 1
  %a.add = add i8 %a.load, 1
  ; SM60: st.volatile.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  ; SM70: st.mmio.relaxed.sys.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i8 %a.add, ptr addrspace(1) %a unordered, align 1

  ; SM60: ld.volatile.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.mmio.relaxed.sys.global.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic volatile i16, ptr addrspace(1) %b unordered, align 2
  %b.add = add i16 %b.load, 1
  ; SM60: st.volatile.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  ; SM70: st.mmio.relaxed.sys.global.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic volatile i16 %b.add, ptr addrspace(1) %b unordered, align 2

  ; SM60: ld.volatile.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.mmio.relaxed.sys.global.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic volatile i32, ptr addrspace(1) %c unordered, align 4
  %c.add = add i32 %c.load, 1
  ; SM60: st.volatile.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  ; SM70: st.mmio.relaxed.sys.global.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic volatile i32 %c.add, ptr addrspace(1) %c unordered, align 4

  ; SM60: ld.volatile.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.mmio.relaxed.sys.global.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic volatile i64, ptr addrspace(1) %d unordered, align 8
  %d.add = add i64 %d.load, 1
  ; SM60: st.volatile.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  ; SM70: st.mmio.relaxed.sys.global.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic volatile i64 %d.add, ptr addrspace(1) %d unordered, align 8

  ; SM60: ld.volatile.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.mmio.relaxed.sys.global.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic volatile float, ptr addrspace(1) %e unordered, align 4
  %e.add = fadd float %e.load, 1.0
  ; SM60: st.volatile.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  ; SM70: st.mmio.relaxed.sys.global.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic volatile float %e.add, ptr addrspace(1) %e unordered, align 4

  ; SM60: ld.volatile.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.mmio.relaxed.sys.global.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic volatile double, ptr addrspace(1) %e unordered, align 8
  %f.add = fadd double %f.load, 1.
  ; SM60: st.volatile.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  ; SM70: st.mmio.relaxed.sys.global.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  store atomic volatile double %f.add, ptr addrspace(1) %e unordered, align 8

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

  ; CHECK: ld.shared.v2.u8 {%rs{{[0-9]+}}, %rs{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %h.load = load <2 x i8>, ptr addrspace(3) %b
  %h.add = add <2 x i8> %h.load, <i8 1, i8 1>
  ; CHECK: st.shared.v2.u8 [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}}
  store <2 x i8> %h.add, ptr addrspace(3) %b

  ; CHECK: ld.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %i.load = load <4 x i8>, ptr addrspace(3) %c
  %i.add = add <4 x i8> %i.load, <i8 1, i8 1, i8 1, i8 1>
  ; CHECK: st.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store <4 x i8> %i.add, ptr addrspace(3) %c

  ; CHECK: ld.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %j.load = load <2 x i16>, ptr addrspace(3) %c
  %j.add = add <2 x i16> %j.load, <i16 1, i16 1>
  ; CHECK: st.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store <2 x i16> %j.add, ptr addrspace(3) %c

  ; CHECK: ld.shared.v4.u16 {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %k.load = load <4 x i16>, ptr addrspace(3) %d
  %k.add = add <4 x i16> %k.load, <i16 1, i16 1, i16 1, i16 1>
  ; CHECK: st.shared.v4.u16 [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}}
  store <4 x i16> %k.add, ptr addrspace(3) %d

  ; CHECK: ld.shared.v2.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %l.load = load <2 x i32>, ptr addrspace(3) %d
  %l.add = add <2 x i32> %l.load, <i32 1, i32 1>
  ; CHECK: st.shared.v2.u32 [%rd{{[0-9]+}}], {%r{{[0-9]+}}, %r{{[0-9]+}}}
  store <2 x i32> %l.add, ptr addrspace(3) %d

  ; CHECK: ld.shared.v4.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %m.load = load <4 x i32>, ptr addrspace(3) %d
  %m.add = add <4 x i32> %m.load, <i32 1, i32 1, i32 1, i32 1>
  ; CHECK: st.shared.v4.u32 [%rd{{[0-9]+}}], {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}
  store <4 x i32> %m.add, ptr addrspace(3) %d

  ; CHECK: ld.shared.v2.u64 {%rd{{[0-9]+}}, %rd{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %n.load = load <2 x i64>, ptr addrspace(3) %d
  %n.add = add <2 x i64> %n.load, <i64 1, i64 1>
  ; CHECK: st.shared.v2.u64 [%rd{{[0-9]+}}], {%rd{{[0-9]+}}, %rd{{[0-9]+}}}
  store <2 x i64> %n.add, ptr addrspace(3) %d

  ; CHECK: ld.shared.v2.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %o.load = load <2 x float>, ptr addrspace(3) %d
  %o.add = fadd <2 x float> %o.load, <float 1., float 1.>
  ; CHECK: st.shared.v2.f32 [%rd{{[0-9]+}}], {%f{{[0-9]+}}, %f{{[0-9]+}}}
  store <2 x float> %o.add, ptr addrspace(3) %d

  ; CHECK: ld.shared.v4.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %p.load = load <4 x float>, ptr addrspace(3) %d
  %p.add = fadd <4 x float> %p.load, <float 1., float 1., float 1., float 1.>
  ; CHECK: st.shared.v4.f32 [%rd{{[0-9]+}}], {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}
  store <4 x float> %p.add, ptr addrspace(3) %d

  ; CHECK: ld.shared.v2.f64 {%fd{{[0-9]+}}, %fd{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %q.load = load <2 x double>, ptr addrspace(3) %d
  %q.add = fadd <2 x double> %q.load, <double 1., double 1.>
  ; CHECK: st.shared.v2.f64 [%rd{{[0-9]+}}], {%fd{{[0-9]+}}, %fd{{[0-9]+}}}
  store <2 x double> %q.add, ptr addrspace(3) %d

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

  ; CHECK: ld.volatile.shared.v2.u8 {%rs{{[0-9]+}}, %rs{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %h.load = load volatile <2 x i8>, ptr addrspace(3) %b
  %h.add = add <2 x i8> %h.load, <i8 1, i8 1>
  ; CHECK: st.volatile.shared.v2.u8 [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}}
  store volatile <2 x i8> %h.add, ptr addrspace(3) %b

  ; CHECK: ld.volatile.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %i.load = load volatile <4 x i8>, ptr addrspace(3) %c
  %i.add = add <4 x i8> %i.load, <i8 1, i8 1, i8 1, i8 1>
  ; CHECK: st.volatile.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store volatile <4 x i8> %i.add, ptr addrspace(3) %c

  ; CHECK: ld.volatile.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %j.load = load volatile <2 x i16>, ptr addrspace(3) %c
  %j.add = add <2 x i16> %j.load, <i16 1, i16 1>
  ; CHECK: st.volatile.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store volatile <2 x i16> %j.add, ptr addrspace(3) %c

  ; CHECK: ld.volatile.shared.v4.u16 {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %k.load = load volatile <4 x i16>, ptr addrspace(3) %d
  %k.add = add <4 x i16> %k.load, <i16 1, i16 1, i16 1, i16 1>
  ; CHECK: st.volatile.shared.v4.u16 [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}}
  store volatile <4 x i16> %k.add, ptr addrspace(3) %d

  ; CHECK: ld.volatile.shared.v2.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %l.load = load volatile <2 x i32>, ptr addrspace(3) %d
  %l.add = add <2 x i32> %l.load, <i32 1, i32 1>
  ; CHECK: st.volatile.shared.v2.u32 [%rd{{[0-9]+}}], {%r{{[0-9]+}}, %r{{[0-9]+}}}
  store volatile <2 x i32> %l.add, ptr addrspace(3) %d

  ; CHECK: ld.volatile.shared.v4.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %m.load = load volatile <4 x i32>, ptr addrspace(3) %d
  %m.add = add <4 x i32> %m.load, <i32 1, i32 1, i32 1, i32 1>
  ; CHECK: st.volatile.shared.v4.u32 [%rd{{[0-9]+}}], {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}
  store volatile <4 x i32> %m.add, ptr addrspace(3) %d

  ; CHECK: ld.volatile.shared.v2.u64 {%rd{{[0-9]+}}, %rd{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %n.load = load volatile <2 x i64>, ptr addrspace(3) %d
  %n.add = add <2 x i64> %n.load, <i64 1, i64 1>
  ; CHECK: st.volatile.shared.v2.u64 [%rd{{[0-9]+}}], {%rd{{[0-9]+}}, %rd{{[0-9]+}}}
  store volatile <2 x i64> %n.add, ptr addrspace(3) %d

  ; CHECK: ld.volatile.shared.v2.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %o.load = load volatile <2 x float>, ptr addrspace(3) %d
  %o.add = fadd <2 x float> %o.load, <float 1., float 1.>
  ; CHECK: st.volatile.shared.v2.f32 [%rd{{[0-9]+}}], {%f{{[0-9]+}}, %f{{[0-9]+}}}
  store volatile <2 x float> %o.add, ptr addrspace(3) %d

  ; CHECK: ld.volatile.shared.v4.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %p.load = load volatile <4 x float>, ptr addrspace(3) %d
  %p.add = fadd <4 x float> %p.load, <float 1., float 1., float 1., float 1.>
  ; CHECK: st.volatile.shared.v4.f32 [%rd{{[0-9]+}}], {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}
  store volatile <4 x float> %p.add, ptr addrspace(3) %d

  ; CHECK: ld.volatile.shared.v2.f64 {%fd{{[0-9]+}}, %fd{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %q.load = load volatile <2 x double>, ptr addrspace(3) %d
  %q.add = fadd <2 x double> %q.load, <double 1., double 1.>
  ; CHECK: st.volatile.shared.v2.f64 [%rd{{[0-9]+}}], {%fd{{[0-9]+}}, %fd{{[0-9]+}}}
  store volatile <2 x double> %q.add, ptr addrspace(3) %d

  ret void
}

; CHECK-LABEL: shared_monotonic
define void @shared_monotonic(ptr addrspace(3) %a, ptr addrspace(3) %b, ptr addrspace(3) %c, ptr addrspace(3) %d, ptr addrspace(3) %e) local_unnamed_addr {
  ; TODO: optimize .sys.shared to .cta.shared or .cluster.shared.

  ; SM60: ld.volatile.shared.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.shared.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(3) %a monotonic, align 1
  %a.add = add i8 %a.load, 1
  ; SM60: st.volatile.shared.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  ; SM70: st.relaxed.sys.shared.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(3) %a monotonic, align 1

  ; SM60: ld.volatile.shared.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.shared.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(3) %b monotonic, align 2
  %b.add = add i16 %b.load, 1
  ; SM60: st.volatile.shared.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  ; SM70: st.relaxed.sys.shared.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(3) %b monotonic, align 2

  ; SM60: ld.volatile.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(3) %c monotonic, align 4
  %c.add = add i32 %c.load, 1
  ; SM60: st.volatile.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  ; SM70: st.relaxed.sys.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(3) %c monotonic, align 4

  ; SM60: ld.volatile.shared.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.shared.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(3) %d monotonic, align 8
  %d.add = add i64 %d.load, 1
  ; SM60: st.volatile.shared.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  ; SM70: st.relaxed.sys.shared.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(3) %d monotonic, align 8

  ; SM60: ld.volatile.shared.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.shared.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(3) %e monotonic, align 4
  %e.add = fadd float %e.load, 1.0
  ; SM60: st.volatile.shared.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  ; SM70: st.relaxed.sys.shared.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(3) %e monotonic, align 4

  ; SM60: ld.volatile.shared.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.shared.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(3) %e monotonic, align 8
  %f.add = fadd double %f.load, 1.
  ; SM60: st.volatile.shared.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  ; SM70: st.relaxed.sys.shared.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
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

; CHECK-LABEL: shared_unordered
define void @shared_unordered(ptr addrspace(3) %a, ptr addrspace(3) %b, ptr addrspace(3) %c, ptr addrspace(3) %d, ptr addrspace(3) %e) local_unnamed_addr {
  ; TODO: optimize .sys.shared to .cta.shared or .cluster.shared.

  ; SM60: ld.volatile.shared.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.shared.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, ptr addrspace(3) %a unordered, align 1
  %a.add = add i8 %a.load, 1
  ; SM60: st.volatile.shared.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  ; SM70: st.relaxed.sys.shared.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, ptr addrspace(3) %a unordered, align 1

  ; SM60: ld.volatile.shared.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.shared.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, ptr addrspace(3) %b unordered, align 2
  %b.add = add i16 %b.load, 1
  ; SM60: st.volatile.shared.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  ; SM70: st.relaxed.sys.shared.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, ptr addrspace(3) %b unordered, align 2

  ; SM60: ld.volatile.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.shared.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, ptr addrspace(3) %c unordered, align 4
  %c.add = add i32 %c.load, 1
  ; SM60: st.volatile.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  ; SM70: st.relaxed.sys.shared.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, ptr addrspace(3) %c unordered, align 4

  ; SM60: ld.volatile.shared.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.shared.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, ptr addrspace(3) %d unordered, align 8
  %d.add = add i64 %d.load, 1
  ; SM60: st.volatile.shared.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  ; SM70: st.relaxed.sys.shared.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, ptr addrspace(3) %d unordered, align 8

  ; SM60: ld.volatile.shared.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.shared.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
  %e.load = load atomic float, ptr addrspace(3) %e unordered, align 4
  %e.add = fadd float %e.load, 1.0
  ; SM60: st.volatile.shared.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  ; SM70: st.relaxed.sys.shared.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
  store atomic float %e.add, ptr addrspace(3) %e unordered, align 4

  ; SM60: ld.volatile.shared.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  ; SM70: ld.relaxed.sys.shared.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %f.load = load atomic double, ptr addrspace(3) %e unordered, align 8
  %f.add = fadd double %f.load, 1.
  ; SM60: st.volatile.shared.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
  ; SM70: st.relaxed.sys.shared.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
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

  ; CHECK: ld.local.v2.u8 {%rs{{[0-9]+}}, %rs{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %h.load = load <2 x i8>, ptr addrspace(5) %b
  %h.add = add <2 x i8> %h.load, <i8 1, i8 1>
  ; CHECK: st.local.v2.u8 [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}}
  store <2 x i8> %h.add, ptr addrspace(5) %b

  ; CHECK: ld.local.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %i.load = load <4 x i8>, ptr addrspace(5) %c
  %i.add = add <4 x i8> %i.load, <i8 1, i8 1, i8 1, i8 1>
  ; CHECK: st.local.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store <4 x i8> %i.add, ptr addrspace(5) %c

  ; CHECK: ld.local.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %j.load = load <2 x i16>, ptr addrspace(5) %c
  %j.add = add <2 x i16> %j.load, <i16 1, i16 1>
  ; CHECK: st.local.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store <2 x i16> %j.add, ptr addrspace(5) %c

  ; CHECK: ld.local.v4.u16 {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %k.load = load <4 x i16>, ptr addrspace(5) %d
  %k.add = add <4 x i16> %k.load, <i16 1, i16 1, i16 1, i16 1>
  ; CHECK: st.local.v4.u16 [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}}
  store <4 x i16> %k.add, ptr addrspace(5) %d

  ; CHECK: ld.local.v2.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %l.load = load <2 x i32>, ptr addrspace(5) %d
  %l.add = add <2 x i32> %l.load, <i32 1, i32 1>
  ; CHECK: st.local.v2.u32 [%rd{{[0-9]+}}], {%r{{[0-9]+}}, %r{{[0-9]+}}}
  store <2 x i32> %l.add, ptr addrspace(5) %d

  ; CHECK: ld.local.v4.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %m.load = load <4 x i32>, ptr addrspace(5) %d
  %m.add = add <4 x i32> %m.load, <i32 1, i32 1, i32 1, i32 1>
  ; CHECK: st.local.v4.u32 [%rd{{[0-9]+}}], {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}
  store <4 x i32> %m.add, ptr addrspace(5) %d

  ; CHECK: ld.local.v2.u64 {%rd{{[0-9]+}}, %rd{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %n.load = load <2 x i64>, ptr addrspace(5) %d
  %n.add = add <2 x i64> %n.load, <i64 1, i64 1>
  ; CHECK: st.local.v2.u64 [%rd{{[0-9]+}}], {%rd{{[0-9]+}}, %rd{{[0-9]+}}}
  store <2 x i64> %n.add, ptr addrspace(5) %d

  ; CHECK: ld.local.v2.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %o.load = load <2 x float>, ptr addrspace(5) %d
  %o.add = fadd <2 x float> %o.load, <float 1., float 1.>
  ; CHECK: st.local.v2.f32 [%rd{{[0-9]+}}], {%f{{[0-9]+}}, %f{{[0-9]+}}}
  store <2 x float> %o.add, ptr addrspace(5) %d

  ; CHECK: ld.local.v4.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %p.load = load <4 x float>, ptr addrspace(5) %d
  %p.add = fadd <4 x float> %p.load, <float 1., float 1., float 1., float 1.>
  ; CHECK: st.local.v4.f32 [%rd{{[0-9]+}}], {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}
  store <4 x float> %p.add, ptr addrspace(5) %d

  ; CHECK: ld.local.v2.f64 {%fd{{[0-9]+}}, %fd{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %q.load = load <2 x double>, ptr addrspace(5) %d
  %q.add = fadd <2 x double> %q.load, <double 1., double 1.>
  ; CHECK: st.local.v2.f64 [%rd{{[0-9]+}}], {%fd{{[0-9]+}}, %fd{{[0-9]+}}}
  store <2 x double> %q.add, ptr addrspace(5) %d

  ret void
}

; CHECK-LABEL: local_volatile
define void @local_volatile(ptr addrspace(5) %a, ptr addrspace(5) %b, ptr addrspace(5) %c, ptr addrspace(5) %d) local_unnamed_addr {
  ; TODO: generate PTX that preserves Concurrent Forward Progress
  ;       by using volatile operations.

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

  ; CHECK: ld.local.v2.u8 {%rs{{[0-9]+}}, %rs{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %h.load = load volatile <2 x i8>, ptr addrspace(5) %b
  %h.add = add <2 x i8> %h.load, <i8 1, i8 1>
  ; CHECK: st.local.v2.u8 [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}}
  store volatile <2 x i8> %h.add, ptr addrspace(5) %b

  ; CHECK: ld.local.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %i.load = load volatile <4 x i8>, ptr addrspace(5) %c
  %i.add = add <4 x i8> %i.load, <i8 1, i8 1, i8 1, i8 1>
  ; CHECK: st.local.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store volatile <4 x i8> %i.add, ptr addrspace(5) %c

  ; CHECK: ld.local.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %j.load = load volatile <2 x i16>, ptr addrspace(5) %c
  %j.add = add <2 x i16> %j.load, <i16 1, i16 1>
  ; CHECK: st.local.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store volatile <2 x i16> %j.add, ptr addrspace(5) %c

  ; CHECK: ld.local.v4.u16 {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %k.load = load volatile <4 x i16>, ptr addrspace(5) %d
  %k.add = add <4 x i16> %k.load, <i16 1, i16 1, i16 1, i16 1>
  ; CHECK: st.local.v4.u16 [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}}
  store volatile <4 x i16> %k.add, ptr addrspace(5) %d

  ; CHECK: ld.local.v2.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %l.load = load volatile <2 x i32>, ptr addrspace(5) %d
  %l.add = add <2 x i32> %l.load, <i32 1, i32 1>
  ; CHECK: st.local.v2.u32 [%rd{{[0-9]+}}], {%r{{[0-9]+}}, %r{{[0-9]+}}}
  store volatile <2 x i32> %l.add, ptr addrspace(5) %d

  ; CHECK: ld.local.v4.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %m.load = load volatile <4 x i32>, ptr addrspace(5) %d
  %m.add = add <4 x i32> %m.load, <i32 1, i32 1, i32 1, i32 1>
  ; CHECK: st.local.v4.u32 [%rd{{[0-9]+}}], {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}
  store volatile <4 x i32> %m.add, ptr addrspace(5) %d

  ; CHECK: ld.local.v2.u64 {%rd{{[0-9]+}}, %rd{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %n.load = load volatile <2 x i64>, ptr addrspace(5) %d
  %n.add = add <2 x i64> %n.load, <i64 1, i64 1>
  ; CHECK: st.local.v2.u64 [%rd{{[0-9]+}}], {%rd{{[0-9]+}}, %rd{{[0-9]+}}}
  store volatile <2 x i64> %n.add, ptr addrspace(5) %d

  ; CHECK: ld.local.v2.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %o.load = load volatile <2 x float>, ptr addrspace(5) %d
  %o.add = fadd <2 x float> %o.load, <float 1., float 1.>
  ; CHECK: st.local.v2.f32 [%rd{{[0-9]+}}], {%f{{[0-9]+}}, %f{{[0-9]+}}}
  store volatile <2 x float> %o.add, ptr addrspace(5) %d

  ; CHECK: ld.local.v4.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %p.load = load volatile <4 x float>, ptr addrspace(5) %d
  %p.add = fadd <4 x float> %p.load, <float 1., float 1., float 1., float 1.>
  ; CHECK: st.local.v4.f32 [%rd{{[0-9]+}}], {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}
  store volatile <4 x float> %p.add, ptr addrspace(5) %d

  ; CHECK: ld.local.v2.f64 {%fd{{[0-9]+}}, %fd{{[0-9]+}}}, [%rd{{[0-9]+}}]
  %q.load = load volatile <2 x double>, ptr addrspace(5) %d
  %q.add = fadd <2 x double> %q.load, <double 1., double 1.>
  ; CHECK: st.local.v2.f64 [%rd{{[0-9]+}}], {%fd{{[0-9]+}}, %fd{{[0-9]+}}}
  store volatile <2 x double> %q.add, ptr addrspace(5) %d

  ret void
}

; CHECK-LABEL: local_monotonic
define void @local_monotonic(ptr addrspace(5) %a, ptr addrspace(5) %b, ptr addrspace(5) %c, ptr addrspace(5) %d, ptr addrspace(5) %e) local_unnamed_addr {
  ; TODO: generate PTX that preserves Concurrent Forward Progress
  ;       by using PTX atomic operations.

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
  ; TODO: generate PTX that preserves Concurrent Forward Progress
  ;       by generating atomic or volatile operations

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

; TODO: add plain,atomic,volatile,atomic volatile tests
;       for .const and .param statespaces