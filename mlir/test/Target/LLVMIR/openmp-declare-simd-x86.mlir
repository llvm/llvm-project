// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s
//
// For x86 mangling: _ZGV <ISA> <Mask> <VLEN> <ParamAttrs> _ <FunctionName>
//   ISA:  b=SSE, c=AVX, d=AVX2, e=AVX-512
//   Mask: M=inbranch, N=notinbranch, both if unspecified
//   ParamAttrs: v=vector, u=uniform, l=linear, L=linear(val),
//               U=linear(uval), R=linear(ref), sN=var-stride(argN),
//               aN=aligned(N)

module attributes {
  llvm.target_triple = "x86_64-unknown-linux-gnu",
  llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
} {

  llvm.func @add_1(%d: !llvm.ptr) {
    %c32 = llvm.mlir.constant(32 : i64) : i64

    omp.declare_simd linear(%d : !llvm.ptr = %c32 : i64)
    omp.declare_simd simdlen(32) inbranch
    omp.declare_simd notinbranch

    llvm.return
  }

  llvm.func @h_int(%hp: !llvm.ptr, %hp2: !llvm.ptr, %hq: !llvm.ptr,
                   %lin: !llvm.ptr) {
    omp.declare_simd aligned(%hp : !llvm.ptr -> 16 : i64,
                             %hp2 : !llvm.ptr -> 16 : i64)
    llvm.return
  }

  llvm.func @h_float(%hp: !llvm.ptr, %hp2: !llvm.ptr, %hq: !llvm.ptr,
                     %lin: !llvm.ptr) {
    omp.declare_simd aligned(%hp : !llvm.ptr -> 16 : i64,
                             %hp2 : !llvm.ptr -> 16 : i64)
    llvm.return
  }

  llvm.func @VV_add(%this: !llvm.ptr, %a: !llvm.ptr, %b: i32) -> i32 {
    %a_val = llvm.load %a : !llvm.ptr -> i32
    omp.declare_simd uniform(%this : !llvm.ptr, %a : !llvm.ptr)
                     linear(val(%b : i32 = %a_val : i32))
    %r = llvm.mlir.constant(0 : i32) : i32
    llvm.return %r : i32
  }

  llvm.func @VV_taddpf(%this: !llvm.ptr, %a: !llvm.ptr, %b: !llvm.ptr) -> f32 {
    %c40 = llvm.mlir.constant(40 : i64) : i64
    %c4  = llvm.mlir.constant(4 : i64) : i64
    %c32 = llvm.mlir.constant(32 : i64) : i64
    omp.declare_simd aligned(%a : !llvm.ptr -> 16 : i64,
                             %b : !llvm.ptr -> 4 : i64)
                     linear(%this : !llvm.ptr = %c40 : i64,
                            %a : !llvm.ptr = %c4 : i64,
                            ref(%b : !llvm.ptr = %c32 : i64))
    %zero = llvm.mlir.constant(0.0 : f32) : f32
    llvm.return %zero : f32
  }

  llvm.func @VV_tadd(%this: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr) -> i32 {
    %c8 = llvm.mlir.constant(8 : i64) : i64
    // #pragma omp declare simd linear(uval(c) : 8)
    omp.declare_simd linear(uval(%c : !llvm.ptr = %c8 : i64))
    // #pragma omp declare simd aligned(b : 8)
    omp.declare_simd aligned(%b : !llvm.ptr -> 8 : i64)
    %zero = llvm.mlir.constant(0 : i32) : i32
    llvm.return %zero : i32
  }

  // aligned(a:32), aligned(b:16)
  // linear(ref(b):16) -> ref, step=16, ptr rescale: 16*sizeof(float*)=8 -> stride=128
  llvm.func @TVV_taddpf(%this: !llvm.ptr, %a: !llvm.ptr, %b: !llvm.ptr) -> f32 {
    %c128 = llvm.mlir.constant(128 : i64) : i64
    omp.declare_simd aligned(%a : !llvm.ptr -> 32 : i64,
                             %b : !llvm.ptr -> 16 : i64)
                     linear(ref(%b : !llvm.ptr = %c128 : i64))
    %zero = llvm.mlir.constant(0.0 : f32) : f32
    llvm.return %zero : f32
  }

  llvm.func @TVV_tadd(%this: !llvm.ptr, %b: !llvm.ptr) -> i32 {
    omp.declare_simd simdlen(16)
    omp.declare_simd uniform(%this : !llvm.ptr, %b : !llvm.ptr)
    %zero = llvm.mlir.constant(0 : i32) : i32
    llvm.return %zero : i32
  }

  llvm.func @foo_tmpl(%b: !llvm.ptr, %c: !llvm.ptr) {
    %c64 = llvm.mlir.constant(64 : i64) : i64
    omp.declare_simd simdlen(64)
                     aligned(%b : !llvm.ptr -> 128 : i64)
                     linear(uval(%c : !llvm.ptr = %c64 : i64))
    llvm.return
  }

  llvm.func @A_infunc(%this: !llvm.ptr, %a: i32) -> i32 {
    %c8 = llvm.mlir.constant(8 : i32) : i32
    omp.declare_simd linear(%a : i32 = %c8 : i32)
    llvm.return %a : i32
  }

  // linear(a:4) -> a is ptr -> Linear; step=4, ptr rescale: 4*sizeof(float)=4 -> stride=16
  llvm.func @A_outfunc(%this: !llvm.ptr, %a: !llvm.ptr) -> f32 {
    %c16 = llvm.mlir.constant(16 : i64) : i64
    omp.declare_simd linear(%a : !llvm.ptr = %c16 : i64)
    %zero = llvm.mlir.constant(0.0 : f32) : f32
    llvm.return %zero : f32
  }

  llvm.func @bar(%v: !llvm.ptr, %a: !llvm.ptr) -> i32 {
    omp.declare_simd
    omp.declare_simd notinbranch aligned(%a : !llvm.ptr -> 32 : i64)
    %zero = llvm.mlir.constant(0 : i32) : i32
    llvm.return %zero : i32
  }

  llvm.func @baz(%v: !llvm.ptr, %a: !llvm.ptr) -> f32 {
    omp.declare_simd
    omp.declare_simd notinbranch aligned(%a : !llvm.ptr -> 16 : i64)
    %zero = llvm.mlir.constant(0.0 : f32) : f32
    llvm.return %zero : f32
  }

  llvm.func @bay(%v: !llvm.ptr, %a: !llvm.ptr) -> f64 {
    omp.declare_simd
    omp.declare_simd notinbranch aligned(%a : !llvm.ptr -> 16 : i64)
    %zero = llvm.mlir.constant(0.0 : f64) : f64
    llvm.return %zero : f64
  }

  llvm.func @bax(%v: !llvm.ptr, %a: !llvm.ptr, %b: !llvm.ptr) {
    %b_val = llvm.load %b : !llvm.ptr -> i32
    omp.declare_simd
    omp.declare_simd inbranch
                     uniform(%v : !llvm.ptr, %b : !llvm.ptr)
                     linear(%a : !llvm.ptr = %b_val : i32)
    llvm.return
  }

  llvm.func @foo_scalar(%q: !llvm.ptr, %x: f32, %k: i32) -> f32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    omp.declare_simd uniform(%q : !llvm.ptr)
                     aligned(%q : !llvm.ptr -> 16 : i64)
                     linear(%k : i32 = %c1 : i32)
    %zero = llvm.mlir.constant(0.0 : f32) : f32
    llvm.return %zero : f32
  }

  llvm.func @foo_double(%x: f64) -> f64 {
    omp.declare_simd notinbranch
    llvm.return %x : f64
  }

  llvm.func @constlinear(%i: i32) -> f64 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    omp.declare_simd notinbranch linear(%i : i32 = %c1 : i32)
    %zero = llvm.mlir.constant(0.0 : f64) : f64
    llvm.return %zero : f64
  }

  llvm.func @One(%a: !llvm.ptr, %b: !llvm.ptr, %c: i32,
                 %d: !llvm.ptr, %e: !llvm.ptr, %f: i32) -> f64 {
    %c2  = llvm.mlir.constant(2 : i64) : i64
    %c16 = llvm.mlir.constant(16 : i64) : i64
    %c8  = llvm.mlir.constant(8 : i32) : i32
    %c1  = llvm.mlir.constant(1 : i64) : i64
    %c4  = llvm.mlir.constant(4 : i64) : i64
    %c1i = llvm.mlir.constant(1 : i32) : i32
    omp.declare_simd simdlen(4)
       linear(%a : !llvm.ptr = %c2 : i64,
              %b : !llvm.ptr = %c16 : i64,
              %c : i32 = %c8 : i32,
              %d : !llvm.ptr = %c1 : i64,
              %e : !llvm.ptr = %c4 : i64,
              %f : i32 = %c1i : i32)
    %zero = llvm.mlir.constant(0.0 : f64) : f64
    llvm.return %zero : f64
  }

  llvm.func @Two(%a: !llvm.ptr, %b: !llvm.ptr, %c: i32,
                 %d: !llvm.ptr, %e: !llvm.ptr, %f: i32) -> f64 {
    %c2  = llvm.mlir.constant(2 : i64) : i64
    %c16 = llvm.mlir.constant(16 : i64) : i64
    %c8  = llvm.mlir.constant(8 : i32) : i32
    %c1  = llvm.mlir.constant(1 : i64) : i64
    %c4  = llvm.mlir.constant(4 : i64) : i64
    %c1i = llvm.mlir.constant(1 : i32) : i32
    omp.declare_simd simdlen(4)
       linear(val(%a : !llvm.ptr = %c2 : i64),
              val(%b : !llvm.ptr = %c16 : i64),
              val(%c : i32 = %c8 : i32),
              val(%d : !llvm.ptr = %c1 : i64),
              val(%e : !llvm.ptr = %c4 : i64),
              val(%f : i32 = %c1i : i32))
    %zero = llvm.mlir.constant(0.0 : f64) : f64
    llvm.return %zero : f64
  }

  llvm.func @Three(%a: !llvm.ptr, %b: !llvm.ptr) -> f64 {
    %c2 = llvm.mlir.constant(2 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    omp.declare_simd simdlen(4)
       linear(uval(%a : !llvm.ptr = %c2 : i64),
              uval(%b : !llvm.ptr = %c1 : i64))
    %zero = llvm.mlir.constant(0.0 : f64) : f64
    llvm.return %zero : f64
  }

  llvm.func @Four(%a: !llvm.ptr, %b: !llvm.ptr) -> f64 {
    %c8 = llvm.mlir.constant(8 : i64) : i64
    %c4 = llvm.mlir.constant(4 : i64) : i64
    omp.declare_simd simdlen(4)
       linear(ref(%a : !llvm.ptr = %c8 : i64),
              ref(%b : !llvm.ptr = %c4 : i64))
    %zero = llvm.mlir.constant(0.0 : f64) : f64
    llvm.return %zero : f64
  }

  // ParamAttrs:
  //   a: uniform -> u
  //   b: linear(b:2) -> ptr, no modifier -> Linear, step=2 -> l2
  //   c: linear(c:a) -> ptr, no modifier -> Linear, var_stride=arg0 -> ls0
  //   d: linear(val(d):4) -> LinearVal, step=4 -> L4
  //   e: linear(val(e):a) -> LinearVal, var_stride=arg0 -> Ls0
  //   f: linear(uval(f):8) -> LinearUVal, step=8 -> U8
  //   g: linear(uval(g):a) -> LinearUVal, var_stride=arg0 -> Us0
  //   h: linear(ref(h):32) -> LinearRef, pre-rescaled stride=32 -> R32
  //   i: linear(ref(i):a) -> LinearRef, var_stride=arg0 -> Rs0
  llvm.func @Five(%a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr, %d: !llvm.ptr,
                  %e: !llvm.ptr, %f: !llvm.ptr, %g: !llvm.ptr,
                  %h: !llvm.ptr, %i: !llvm.ptr) -> f64 {
    %c2  = llvm.mlir.constant(2 : i64) : i64
    %c4  = llvm.mlir.constant(4 : i64) : i64
    %c8  = llvm.mlir.constant(8 : i64) : i64
    %c32 = llvm.mlir.constant(32 : i64) : i64
    %a_val = llvm.load %a : !llvm.ptr -> i32
    omp.declare_simd simdlen(4)
       uniform(%a : !llvm.ptr)
       linear(%b : !llvm.ptr = %c2 : i64,
              %c : !llvm.ptr = %a_val : i32,
              val(%d : !llvm.ptr = %c4 : i64),
              val(%e : !llvm.ptr = %a_val : i32),
              uval(%f : !llvm.ptr = %c8 : i64),
              uval(%g : !llvm.ptr = %a_val : i32),
              ref(%h : !llvm.ptr = %c32 : i64),
              ref(%i : !llvm.ptr = %a_val : i32))
    %zero = llvm.mlir.constant(0.0 : f64) : f64
    llvm.return %zero : f64
  }

  //   a: i32, linear(-2) -> l, step=-2 -> ln2
  //   b: ptr, linear(-32) -> l, step=-32 -> ln32
  //   c: ptr, uval(-4) -> U, step=-4 -> Un4
  //   d: ptr, ref(-128) -> R, step=-128 -> Rn128
  //   e: i8, linear(-1) -> l, step=-1 -> ln1
  //   f: ptr, linear(-1) -> l, step=-1 -> ln1
  //   g: i16, linear(0) -> l, step=0 -> l0
  llvm.func @Six(%a: i32, %b: !llvm.ptr, %c: !llvm.ptr, %d: !llvm.ptr,
                 %e: i8, %f: !llvm.ptr, %g: i16) -> f64 {
    %cn2   = llvm.mlir.constant(-2 : i32) : i32
    %cn32  = llvm.mlir.constant(-32 : i64) : i64
    %cn4   = llvm.mlir.constant(-4 : i64) : i64
    %cn128 = llvm.mlir.constant(-128 : i64) : i64
    %cn1i  = llvm.mlir.constant(-1 : i32) : i32
    %cn1   = llvm.mlir.constant(-1 : i64) : i64
    %c0    = llvm.mlir.constant(0 : i32) : i32
    omp.declare_simd simdlen(4)
       linear(%a : i32 = %cn2 : i32,
              %b : !llvm.ptr = %cn32 : i64,
              uval(%c : !llvm.ptr = %cn4 : i64),
              ref(%d : !llvm.ptr = %cn128 : i64),
              %e : i8 = %cn1i : i32,
              %f : !llvm.ptr = %cn1 : i64,
              %g : i16 = %c0 : i32)
    %zero = llvm.mlir.constant(0.0 : f64) : f64
    llvm.return %zero : f64
  }
}

// --- add_1: three declare_simd ops ---
//
// linear(d:32): CDT=int(32), VLEN=reg/32
//   b: VLEN=128/32=4, c: VLEN=256/32=8, d: VLEN=256/32=8, e: VLEN=512/32=16
//   Ptr linear -> Linear -> l32
// inbranch simdlen(32): M only
// notinbranch: N only, CDT=int(VLEN varies)
//
// CHECK-DAG: "_ZGVbM4l32_add_1"
// CHECK-DAG: "_ZGVbN4l32_add_1"
// CHECK-DAG: "_ZGVcM8l32_add_1"
// CHECK-DAG: "_ZGVcN8l32_add_1"
// CHECK-DAG: "_ZGVdM8l32_add_1"
// CHECK-DAG: "_ZGVdN8l32_add_1"
// CHECK-DAG: "_ZGVeM16l32_add_1"
// CHECK-DAG: "_ZGVeN16l32_add_1"

// CHECK-DAG: "_ZGVbM32v_add_1"
// CHECK-DAG: "_ZGVcM32v_add_1"
// CHECK-DAG: "_ZGVdM32v_add_1"
// CHECK-DAG: "_ZGVeM32v_add_1"

// CHECK-DAG: "_ZGVbN2v_add_1"
// CHECK-DAG: "_ZGVcN4v_add_1"
// CHECK-DAG: "_ZGVdN4v_add_1"
// CHECK-DAG: "_ZGVeN8v_add_1"

// --- h_int ---
// aligned(hp:16, hp2:16), CDT=ptr(64-bit) -> VLEN: b=2,c=4,d=4,e=8
// CHECK-DAG: "_ZGVbM2va16va16vv_h_int"
// CHECK-DAG: "_ZGVbN2va16va16vv_h_int"
// CHECK-DAG: "_ZGVcM4va16va16vv_h_int"
// CHECK-DAG: "_ZGVcN4va16va16vv_h_int"
// CHECK-DAG: "_ZGVdM4va16va16vv_h_int"
// CHECK-DAG: "_ZGVdN4va16va16vv_h_int"
// CHECK-DAG: "_ZGVeM8va16va16vv_h_int"
// CHECK-DAG: "_ZGVeN8va16va16vv_h_int"

// --- h_float ---
// aligned(hp:16, hp2:16), CDT=ptr(64-bit) -> VLEN: b=2,c=4,d=4,e=8
// CHECK-DAG: "_ZGVbM2va16va16vv_h_float"
// CHECK-DAG: "_ZGVbN2va16va16vv_h_float"
// CHECK-DAG: "_ZGVcM4va16va16vv_h_float"
// CHECK-DAG: "_ZGVcN4va16va16vv_h_float"
// CHECK-DAG: "_ZGVdM4va16va16vv_h_float"
// CHECK-DAG: "_ZGVdN4va16va16vv_h_float"
// CHECK-DAG: "_ZGVeM8va16va16vv_h_float"
// CHECK-DAG: "_ZGVeN8va16va16vv_h_float"

// --- VV_add: uniform(this,a), linear(val(b):var_stride=a=arg1) ---
// val on i32 (non-pointer) -> Linear (l)
// CDT = return i32 (32-bit) -> VLEN: b=4, c=8, d=8, e=16
// CHECK-DAG: "_ZGVbM4uuls1_VV_add"
// CHECK-DAG: "_ZGVbN4uuls1_VV_add"
// CHECK-DAG: "_ZGVcM8uuls1_VV_add"
// CHECK-DAG: "_ZGVcN8uuls1_VV_add"
// CHECK-DAG: "_ZGVdM8uuls1_VV_add"
// CHECK-DAG: "_ZGVdN8uuls1_VV_add"
// CHECK-DAG: "_ZGVeM16uuls1_VV_add"
// CHECK-DAG: "_ZGVeN16uuls1_VV_add"

// --- VV_taddpf ---
// linear(this) -> this is ptr -> Linear; step=1, ptr rescale: 1*sizeof(VV)=40 -> stride=40
// linear(a)    -> a is ptr -> Linear; step=1, ptr rescale: 1*sizeof(float)=4 -> stride=4
// linear(ref(b):4) -> LinearRef; step=4, ptr rescale: 4*sizeof(float*)=8 -> stride=32
// aligned(a) -> default=16; aligned(b:4) -> 4
// CDT = return f32 (32-bit) -> VLEN: b=4, c=8, d=8, e=16
// CHECK-DAG: "_ZGVbM4l40l4a16R32a4_VV_taddpf"
// CHECK-DAG: "_ZGVbN4l40l4a16R32a4_VV_taddpf"
// CHECK-DAG: "_ZGVcM8l40l4a16R32a4_VV_taddpf"
// CHECK-DAG: "_ZGVcN8l40l4a16R32a4_VV_taddpf"
// CHECK-DAG: "_ZGVdM8l40l4a16R32a4_VV_taddpf"
// CHECK-DAG: "_ZGVdN8l40l4a16R32a4_VV_taddpf"
// CHECK-DAG: "_ZGVeM16l40l4a16R32a4_VV_taddpf"
// CHECK-DAG: "_ZGVeN16l40l4a16R32a4_VV_taddpf"

// --- VV_tadd: ---
// linear(uval(c):8) -> v v U8
// aligned(b:8) -> v va8 v
// CDT = return i32 -> VLEN: b=4, c=8, d=8, e=16
// CHECK-DAG: "_ZGVbM4vvU8_VV_tadd"
// CHECK-DAG: "_ZGVbN4vvU8_VV_tadd"
// CHECK-DAG: "_ZGVcM8vvU8_VV_tadd"
// CHECK-DAG: "_ZGVcN8vvU8_VV_tadd"
// CHECK-DAG: "_ZGVdM8vvU8_VV_tadd"
// CHECK-DAG: "_ZGVdN8vvU8_VV_tadd"
// CHECK-DAG: "_ZGVeM16vvU8_VV_tadd"
// CHECK-DAG: "_ZGVeN16vvU8_VV_tadd"

// CHECK-DAG: "_ZGVbM4vva8v_VV_tadd"
// CHECK-DAG: "_ZGVbN4vva8v_VV_tadd"
// CHECK-DAG: "_ZGVcM8vva8v_VV_tadd"
// CHECK-DAG: "_ZGVcN8vva8v_VV_tadd"
// CHECK-DAG: "_ZGVdM8vva8v_VV_tadd"
// CHECK-DAG: "_ZGVdN8vva8v_VV_tadd"
// CHECK-DAG: "_ZGVeM16vva8v_VV_tadd"
// CHECK-DAG: "_ZGVeN16vva8v_VV_tadd"

// --- TVV_taddpf ---
// aligned(a:32), aligned(b:16), ref(b:128)
// CDT = return f32 -> VLEN: b=4, c=8, d=8, e=16
// CHECK-DAG: "_ZGVbM4vva32R128a16_TVV_taddpf"
// CHECK-DAG: "_ZGVbN4vva32R128a16_TVV_taddpf"
// CHECK-DAG: "_ZGVcM8vva32R128a16_TVV_taddpf"
// CHECK-DAG: "_ZGVcN8vva32R128a16_TVV_taddpf"
// CHECK-DAG: "_ZGVdM8vva32R128a16_TVV_taddpf"
// CHECK-DAG: "_ZGVdN8vva32R128a16_TVV_taddpf"
// CHECK-DAG: "_ZGVeM16vva32R128a16_TVV_taddpf"
// CHECK-DAG: "_ZGVeN16vva32R128a16_TVV_taddpf"

// --- TVV_tadd: ---
// simdlen(16) -> VLEN=16, all vector -> vv
// uniform(this, b) -> uu
// CDT = return i32 -> VLEN: b=4, c=8, d=8, e=16
// CHECK-DAG: "_ZGVbM4uu_TVV_tadd"
// CHECK-DAG: "_ZGVbN4uu_TVV_tadd"
// CHECK-DAG: "_ZGVcM8uu_TVV_tadd"
// CHECK-DAG: "_ZGVcN8uu_TVV_tadd"
// CHECK-DAG: "_ZGVdM8uu_TVV_tadd"
// CHECK-DAG: "_ZGVdN8uu_TVV_tadd"
// CHECK-DAG: "_ZGVeM16uu_TVV_tadd"
// CHECK-DAG: "_ZGVeN16uu_TVV_tadd"

// CHECK-DAG: "_ZGVbM16vv_TVV_tadd"
// CHECK-DAG: "_ZGVbN16vv_TVV_tadd"
// CHECK-DAG: "_ZGVcM16vv_TVV_tadd"
// CHECK-DAG: "_ZGVcN16vv_TVV_tadd"
// CHECK-DAG: "_ZGVdM16vv_TVV_tadd"
// CHECK-DAG: "_ZGVdN16vv_TVV_tadd"
// CHECK-DAG: "_ZGVeM16vv_TVV_tadd"
// CHECK-DAG: "_ZGVeN16vv_TVV_tadd"

// --- foo_tmpl: ---
// simdlen(64), aligned(b:128), uval(c:64)
// CHECK-DAG: "_ZGVbM64va128U64_foo_tmpl"
// CHECK-DAG: "_ZGVbN64va128U64_foo_tmpl"
// CHECK-DAG: "_ZGVcM64va128U64_foo_tmpl"
// CHECK-DAG: "_ZGVcN64va128U64_foo_tmpl"
// CHECK-DAG: "_ZGVdM64va128U64_foo_tmpl"
// CHECK-DAG: "_ZGVdN64va128U64_foo_tmpl"
// CHECK-DAG: "_ZGVeM64va128U64_foo_tmpl"
// CHECK-DAG: "_ZGVeN64va128U64_foo_tmpl"

// --- A_infunc: ---
// linear(a:8), a is i32 -> Linear
// CDT = return i32 -> VLEN: b=4, c=8, d=8, e=16
// CHECK-DAG: "_ZGVbM4vl8_A_infunc"
// CHECK-DAG: "_ZGVbN4vl8_A_infunc"
// CHECK-DAG: "_ZGVcM8vl8_A_infunc"
// CHECK-DAG: "_ZGVcN8vl8_A_infunc"
// CHECK-DAG: "_ZGVdM8vl8_A_infunc"
// CHECK-DAG: "_ZGVdN8vl8_A_infunc"
// CHECK-DAG: "_ZGVeM16vl8_A_infunc"
// CHECK-DAG: "_ZGVeN16vl8_A_infunc"

// --- A_outfunc: ---
// linear(a:16), a is ptr -> Linear
// CDT = return f32 -> VLEN: b=4, c=8, d=8, e=16
// CHECK-DAG: "_ZGVbM4vl16_A_outfunc"
// CHECK-DAG: "_ZGVbN4vl16_A_outfunc"
// CHECK-DAG: "_ZGVcM8vl16_A_outfunc"
// CHECK-DAG: "_ZGVcN8vl16_A_outfunc"
// CHECK-DAG: "_ZGVdM8vl16_A_outfunc"
// CHECK-DAG: "_ZGVdN8vl16_A_outfunc"
// CHECK-DAG: "_ZGVeM16vl16_A_outfunc"
// CHECK-DAG: "_ZGVeN16vl16_A_outfunc"

// --- bar: two declare_simd ---
// all vector -> vv, CDT=return i32, VLEN: b=4,c=8,d=8,e=16
// CHECK-DAG: "_ZGVbM4vv_bar"
// CHECK-DAG: "_ZGVbN4vv_bar"
// CHECK-DAG: "_ZGVcM8vv_bar"
// CHECK-DAG: "_ZGVcN8vv_bar"
// CHECK-DAG: "_ZGVdM8vv_bar"
// CHECK-DAG: "_ZGVdN8vv_bar"
// CHECK-DAG: "_ZGVeM16vv_bar"
// CHECK-DAG: "_ZGVeN16vv_bar"

// notinbranch, aligned(a:32)
// CHECK-DAG: "_ZGVbN4vva32_bar"
// CHECK-DAG: "_ZGVcN8vva32_bar"
// CHECK-DAG: "_ZGVdN8vva32_bar"
// CHECK-DAG: "_ZGVeN16vva32_bar"

// --- baz: ---
// CDT=return f32 -> VLEN: b=4,c=8,d=8,e=16
// CHECK-DAG: "_ZGVbM4vv_baz"
// CHECK-DAG: "_ZGVbN4vv_baz"
// CHECK-DAG: "_ZGVcM8vv_baz"
// CHECK-DAG: "_ZGVcN8vv_baz"
// CHECK-DAG: "_ZGVdM8vv_baz"
// CHECK-DAG: "_ZGVdN8vv_baz"
// CHECK-DAG: "_ZGVeM16vv_baz"
// CHECK-DAG: "_ZGVeN16vv_baz"
// CHECK-DAG: "_ZGVbN4vva16_baz"
// CHECK-DAG: "_ZGVcN8vva16_baz"
// CHECK-DAG: "_ZGVdN8vva16_baz"
// CHECK-DAG: "_ZGVeN16vva16_baz"

// --- bay: ---
// CDT=f64(64-bit) -> VLEN: b=2,c=4,d=4,e=8
// CHECK-DAG: "_ZGVbM2vv_bay"
// CHECK-DAG: "_ZGVbN2vv_bay"
// CHECK-DAG: "_ZGVcM4vv_bay"
// CHECK-DAG: "_ZGVcN4vv_bay"
// CHECK-DAG: "_ZGVdM4vv_bay"
// CHECK-DAG: "_ZGVdN4vv_bay"
// CHECK-DAG: "_ZGVeM8vv_bay"
// CHECK-DAG: "_ZGVeN8vv_bay"
// CHECK-DAG: "_ZGVbN2vva16_bay"
// CHECK-DAG: "_ZGVcN4vva16_bay"
// CHECK-DAG: "_ZGVdN4vva16_bay"
// CHECK-DAG: "_ZGVeN8vva16_bay"

// --- bax: inbranch ---
// all vector -> vvv, CDT=ptr(64-bit), VLEN: b=2,c=4,d=4,e=8
// no vector params -> CDT=int(32) -> VLEN: b=4,c=8,d=8,e=16
// CHECK-DAG: "_ZGVbM2vvv_bax"
// CHECK-DAG: "_ZGVbN2vvv_bax"
// CHECK-DAG: "_ZGVcM4vvv_bax"
// CHECK-DAG: "_ZGVcN4vvv_bax"
// CHECK-DAG: "_ZGVdM4vvv_bax"
// CHECK-DAG: "_ZGVdN4vvv_bax"
// CHECK-DAG: "_ZGVeM8vvv_bax"
// CHECK-DAG: "_ZGVeN8vvv_bax"

// inbranch, uniform(v,b), linear(a:b) -> a is ptr -> Linear var-stride
// ParamAttrs: u ls2 u -> uls2u
// CHECK-DAG: "_ZGVbM4uls2u_bax"
// CHECK-DAG: "_ZGVcM8uls2u_bax"
// CHECK-DAG: "_ZGVdM8uls2u_bax"
// CHECK-DAG: "_ZGVeM16uls2u_bax"

// --- foo_scalar: ---
// ParamAttrs: [0]=uniform+aligned(16), [1]=vector, [2]=linear(k:1)
// k is i32 -> Linear (non-ptr), step=1
// uniform(q)+aligned(q:16) + linear(k:1)
// CDT = return f32 -> VLEN: b=4,c=8,d=8,e=16
// CHECK-DAG: "_ZGVbM4ua16vl_foo_scalar"
// CHECK-DAG: "_ZGVbN4ua16vl_foo_scalar"
// CHECK-DAG: "_ZGVcM8ua16vl_foo_scalar"
// CHECK-DAG: "_ZGVcN8ua16vl_foo_scalar"
// CHECK-DAG: "_ZGVdM8ua16vl_foo_scalar"
// CHECK-DAG: "_ZGVdN8ua16vl_foo_scalar"
// CHECK-DAG: "_ZGVeM16ua16vl_foo_scalar"
// CHECK-DAG: "_ZGVeN16ua16vl_foo_scalar"

// --- foo_double: ---
// CDT = f64 (64-bit) -> VLEN: b=2,c=4,d=4,e=8
// CHECK-DAG: "_ZGVbN2v_foo_double"
// CHECK-DAG: "_ZGVcN4v_foo_double"
// CHECK-DAG: "_ZGVdN4v_foo_double"
// CHECK-DAG: "_ZGVeN8v_foo_double"

// --- constlinear: notinbranch ---
// linear(i:1), i is i32 -> Linear
// CDT = f64 (64-bit) -> VLEN: b=2,c=4,d=4,e=8
// CHECK-DAG: "_ZGVbN2l_constlinear"
// CHECK-DAG: "_ZGVcN4l_constlinear"
// CHECK-DAG: "_ZGVdN4l_constlinear"
// CHECK-DAG: "_ZGVeN8l_constlinear"

// --- One: ---
// linear() without modifier, simdlen(4)
// ptr->Linear(l), i32->Linear(l)
// CHECK-DAG: "_ZGVbM4l2l16l8ll4l_One"
// CHECK-DAG: "_ZGVbN4l2l16l8ll4l_One"
// CHECK-DAG: "_ZGVcM4l2l16l8ll4l_One"
// CHECK-DAG: "_ZGVcN4l2l16l8ll4l_One"
// CHECK-DAG: "_ZGVdM4l2l16l8ll4l_One"
// CHECK-DAG: "_ZGVdN4l2l16l8ll4l_One"
// CHECK-DAG: "_ZGVeM4l2l16l8ll4l_One"
// CHECK-DAG: "_ZGVeN4l2l16l8ll4l_One"

// --- Two: ---
// linear(val), simdlen(4)
// val on !llvm.ptr -> L, val on i32 -> l
// CHECK-DAG: "_ZGVbM4L2L16l8LL4l_Two"
// CHECK-DAG: "_ZGVbN4L2L16l8LL4l_Two"
// CHECK-DAG: "_ZGVcM4L2L16l8LL4l_Two"
// CHECK-DAG: "_ZGVcN4L2L16l8LL4l_Two"
// CHECK-DAG: "_ZGVdM4L2L16l8LL4l_Two"
// CHECK-DAG: "_ZGVdN4L2L16l8LL4l_Two"
// CHECK-DAG: "_ZGVeM4L2L16l8LL4l_Two"
// CHECK-DAG: "_ZGVeN4L2L16l8LL4l_Two"

// --- Three: ---
// uval: U2 U, simdlen(4)
// CHECK-DAG: "_ZGVbM4U2U_Three"
// CHECK-DAG: "_ZGVbN4U2U_Three"
// CHECK-DAG: "_ZGVcM4U2U_Three"
// CHECK-DAG: "_ZGVcN4U2U_Three"
// CHECK-DAG: "_ZGVdM4U2U_Three"
// CHECK-DAG: "_ZGVdN4U2U_Three"
// CHECK-DAG: "_ZGVeM4U2U_Three"
// CHECK-DAG: "_ZGVeN4U2U_Three"

// --- Four: ---
// ref, simdlen(4)
// CHECK-DAG: "_ZGVbM4R8R4_Four"
// CHECK-DAG: "_ZGVbN4R8R4_Four"
// CHECK-DAG: "_ZGVcM4R8R4_Four"
// CHECK-DAG: "_ZGVcN4R8R4_Four"
// CHECK-DAG: "_ZGVdM4R8R4_Four"
// CHECK-DAG: "_ZGVdN4R8R4_Four"
// CHECK-DAG: "_ZGVeM4R8R4_Four"
// CHECK-DAG: "_ZGVeN4R8R4_Four"

// --- Five: ---
// all modifiers + var stride, simdlen(4)
// u l2 ls0 L4 Ls0 U8 Us0 R32 Rs0
// CHECK-DAG: "_ZGVbM4ul2ls0L4Ls0U8Us0R32Rs0_Five"
// CHECK-DAG: "_ZGVbN4ul2ls0L4Ls0U8Us0R32Rs0_Five"
// CHECK-DAG: "_ZGVcM4ul2ls0L4Ls0U8Us0R32Rs0_Five"
// CHECK-DAG: "_ZGVcN4ul2ls0L4Ls0U8Us0R32Rs0_Five"
// CHECK-DAG: "_ZGVdM4ul2ls0L4Ls0U8Us0R32Rs0_Five"
// CHECK-DAG: "_ZGVdN4ul2ls0L4Ls0U8Us0R32Rs0_Five"
// CHECK-DAG: "_ZGVeM4ul2ls0L4Ls0U8Us0R32Rs0_Five"
// CHECK-DAG: "_ZGVeN4ul2ls0L4Ls0U8Us0R32Rs0_Five"

// --- Six: ---
// negative strides, simdlen(4)
// ln2 ln32 Un4 Rn128 ln1 ln1 l0
// CHECK-DAG: "_ZGVbM4ln2ln32Un4Rn128ln1ln1l0_Six"
// CHECK-DAG: "_ZGVbN4ln2ln32Un4Rn128ln1ln1l0_Six"
// CHECK-DAG: "_ZGVcM4ln2ln32Un4Rn128ln1ln1l0_Six"
// CHECK-DAG: "_ZGVcN4ln2ln32Un4Rn128ln1ln1l0_Six"
// CHECK-DAG: "_ZGVdM4ln2ln32Un4Rn128ln1ln1l0_Six"
// CHECK-DAG: "_ZGVdN4ln2ln32Un4Rn128ln1ln1l0_Six"
// CHECK-DAG: "_ZGVeM4ln2ln32Un4Rn128ln1ln1l0_Six"
// CHECK-DAG: "_ZGVeN4ln2ln32Un4Rn128ln1ln1l0_Six"
