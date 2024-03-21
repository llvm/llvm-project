; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme < %s | FileCheck %s

; WITH VSCALE RANGE

define i64 @ctz_nxv8i1(<vscale x 8 x i1> %a) #0 {
; CHECK-LABEL: ctz_nxv8i1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    index z0.h, #0, #-1
; CHECK-NEXT:    mov z1.h, p0/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    ptrue p0.h
; CHECK-NEXT:    cnth x9
; CHECK-NEXT:    inch z0.h
; CHECK-NEXT:    and z0.d, z0.d, z1.d
; CHECK-NEXT:    and z0.h, z0.h, #0xff
; CHECK-NEXT:    umaxv h0, p0, z0.h
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w8, w9, w8
; CHECK-NEXT:    and x0, x8, #0xff
; CHECK-NEXT:    ret
  %res = call i64 @llvm.experimental.cttz.elts.i64.nxv8i1(<vscale x 8 x i1> %a, i1 0)
  ret i64 %res
}

define i32 @ctz_nxv32i1(<vscale x 32 x i1> %a) #0 {
; CHECK-LABEL: ctz_nxv32i1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    index z0.h, #0, #-1
; CHECK-NEXT:    cnth x8
; CHECK-NEXT:    punpklo p2.h, p0.b
; CHECK-NEXT:    neg x8, x8
; CHECK-NEXT:    punpklo p3.h, p1.b
; CHECK-NEXT:    rdvl x9, #2
; CHECK-NEXT:    punpkhi p0.h, p0.b
; CHECK-NEXT:    mov z1.h, w8
; CHECK-NEXT:    rdvl x8, #-1
; CHECK-NEXT:    punpkhi p1.h, p1.b
; CHECK-NEXT:    mov z2.h, w8
; CHECK-NEXT:    inch z0.h, all, mul #4
; CHECK-NEXT:    mov z3.h, p2/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    ptrue p2.h
; CHECK-NEXT:    mov z5.h, p3/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    add z1.h, z0.h, z1.h
; CHECK-NEXT:    add z4.h, z0.h, z2.h
; CHECK-NEXT:    mov z6.h, p0/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    mov z7.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    and z0.d, z0.d, z3.d
; CHECK-NEXT:    add z2.h, z1.h, z2.h
; CHECK-NEXT:    and z3.d, z4.d, z5.d
; CHECK-NEXT:    and z1.d, z1.d, z6.d
; CHECK-NEXT:    and z2.d, z2.d, z7.d
; CHECK-NEXT:    umax z0.h, p2/m, z0.h, z3.h
; CHECK-NEXT:    umax z1.h, p2/m, z1.h, z2.h
; CHECK-NEXT:    umax z0.h, p2/m, z0.h, z1.h
; CHECK-NEXT:    umaxv h0, p2, z0.h
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w8, w9, w8
; CHECK-NEXT:    and w0, w8, #0xffff
; CHECK-NEXT:    ret
  %res = call i32 @llvm.experimental.cttz.elts.i32.nxv32i1(<vscale x 32 x i1> %a, i1 0)
  ret i32 %res
}

define i32 @ctz_nxv4i32(<vscale x 4 x i32> %a) #0 {
; CHECK-LABEL: ctz_nxv4i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.s
; CHECK-NEXT:    index z1.s, #0, #-1
; CHECK-NEXT:    cntw x9
; CHECK-NEXT:    incw z1.s
; CHECK-NEXT:    cmpne p1.s, p0/z, z0.s, #0
; CHECK-NEXT:    mov z0.s, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    and z0.d, z1.d, z0.d
; CHECK-NEXT:    and z0.s, z0.s, #0xff
; CHECK-NEXT:    umaxv s0, p0, z0.s
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w8, w9, w8
; CHECK-NEXT:    and w0, w8, #0xff
; CHECK-NEXT:    ret
  %res = call i32 @llvm.experimental.cttz.elts.i32.nxv4i32(<vscale x 4 x i32> %a, i1 0)
  ret i32 %res
}

; VSCALE RANGE, ZERO IS POISON

define i64 @vscale_4096(<vscale x 16 x i8> %a) #1 {
; CHECK-LABEL: vscale_4096:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.b
; CHECK-NEXT:    cntw x8
; CHECK-NEXT:    cnth x9
; CHECK-NEXT:    neg x8, x8
; CHECK-NEXT:    mov z1.s, w8
; CHECK-NEXT:    neg x8, x9
; CHECK-NEXT:    rdvl x9, #1
; CHECK-NEXT:    mov z2.s, w8
; CHECK-NEXT:    cmpne p0.b, p0/z, z0.b, #0
; CHECK-NEXT:    index z0.s, #0, #-1
; CHECK-NEXT:    punpklo p1.h, p0.b
; CHECK-NEXT:    punpkhi p0.h, p0.b
; CHECK-NEXT:    incw z0.s, all, mul #4
; CHECK-NEXT:    add z1.s, z0.s, z1.s
; CHECK-NEXT:    add z5.s, z0.s, z2.s
; CHECK-NEXT:    punpkhi p2.h, p1.b
; CHECK-NEXT:    punpkhi p3.h, p0.b
; CHECK-NEXT:    punpklo p0.h, p0.b
; CHECK-NEXT:    add z2.s, z1.s, z2.s
; CHECK-NEXT:    punpklo p1.h, p1.b
; CHECK-NEXT:    mov z3.s, p2/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    ptrue p2.s
; CHECK-NEXT:    mov z4.s, p3/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    mov z6.s, p0/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    mov z7.s, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    and z1.d, z1.d, z3.d
; CHECK-NEXT:    and z2.d, z2.d, z4.d
; CHECK-NEXT:    and z3.d, z5.d, z6.d
; CHECK-NEXT:    and z0.d, z0.d, z7.d
; CHECK-NEXT:    umax z1.s, p2/m, z1.s, z2.s
; CHECK-NEXT:    umax z0.s, p2/m, z0.s, z3.s
; CHECK-NEXT:    umax z0.s, p2/m, z0.s, z1.s
; CHECK-NEXT:    umaxv s0, p2, z0.s
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w0, w9, w8
; CHECK-NEXT:    ret
  %res = call i64 @llvm.experimental.cttz.elts.i64.nxv16i8(<vscale x 16 x i8> %a, i1 0)
  ret i64 %res
}

define i64 @vscale_4096_poison(<vscale x 16 x i8> %a) #1 {
; CHECK-LABEL: vscale_4096_poison:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.b
; CHECK-NEXT:    cnth x8
; CHECK-NEXT:    rdvl x9, #1
; CHECK-NEXT:    neg x8, x8
; CHECK-NEXT:    mov z1.h, w8
; CHECK-NEXT:    cmpne p0.b, p0/z, z0.b, #0
; CHECK-NEXT:    index z0.h, #0, #-1
; CHECK-NEXT:    punpkhi p1.h, p0.b
; CHECK-NEXT:    punpklo p0.h, p0.b
; CHECK-NEXT:    inch z0.h, all, mul #2
; CHECK-NEXT:    add z1.h, z0.h, z1.h
; CHECK-NEXT:    mov z2.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    mov z3.h, p0/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    ptrue p0.h
; CHECK-NEXT:    and z1.d, z1.d, z2.d
; CHECK-NEXT:    and z0.d, z0.d, z3.d
; CHECK-NEXT:    umax z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT:    umaxv h0, p0, z0.h
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w8, w9, w8
; CHECK-NEXT:    and x0, x8, #0xffff
; CHECK-NEXT:    ret
  %res = call i64 @llvm.experimental.cttz.elts.i64.nxv16i8(<vscale x 16 x i8> %a, i1 1)
  ret i64 %res
}

; NO VSCALE RANGE

define i32 @ctz_nxv8i1_no_range(<vscale x 8 x i1> %a) {
; CHECK-LABEL: ctz_nxv8i1_no_range:
; CHECK:       // %bb.0:
; CHECK-NEXT:    index z0.s, #0, #-1
; CHECK-NEXT:    punpklo p1.h, p0.b
; CHECK-NEXT:    cntw x8
; CHECK-NEXT:    punpkhi p0.h, p0.b
; CHECK-NEXT:    neg x8, x8
; CHECK-NEXT:    cnth x9
; CHECK-NEXT:    mov z1.s, w8
; CHECK-NEXT:    incw z0.s, all, mul #2
; CHECK-NEXT:    mov z2.s, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    mov z3.s, p0/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    ptrue p0.s
; CHECK-NEXT:    add z1.s, z0.s, z1.s
; CHECK-NEXT:    and z0.d, z0.d, z2.d
; CHECK-NEXT:    and z1.d, z1.d, z3.d
; CHECK-NEXT:    umax z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT:    umaxv s0, p0, z0.s
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w0, w9, w8
; CHECK-NEXT:    ret
  %res = call i32 @llvm.experimental.cttz.elts.i32.nxv8i1(<vscale x 8 x i1> %a, i1 0)
  ret i32 %res
}

; MATCH WITH BRKB + CNTP

define i32 @ctz_nxv16i1(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a) {
; CHECK-LABEL: ctz_nxv16i1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.b
; CHECK-NEXT:    brkb p0.b, p0/z, p1.b
; CHECK-NEXT:    cntp x0, p0, p0.b
; CHECK-NEXT:    // kill: def $w0 killed $w0 killed $x0
; CHECK-NEXT:    ret
  %res = call i32 @llvm.experimental.cttz.elts.i32.nxv16i1(<vscale x 16 x i1> %a, i1 0)
  ret i32 %res
}

define i32 @ctz_nxv16i1_poison(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a) {
; CHECK-LABEL: ctz_nxv16i1_poison:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.b
; CHECK-NEXT:    brkb p0.b, p0/z, p1.b
; CHECK-NEXT:    cntp x0, p0, p0.b
; CHECK-NEXT:    // kill: def $w0 killed $w0 killed $x0
; CHECK-NEXT:    ret
  %res = call i32 @llvm.experimental.cttz.elts.i32.nxv16i1(<vscale x 16 x i1> %a, i1 1)
  ret i32 %res
}

define i32 @ctz_and_nxv16i1(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: ctz_and_nxv16i1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p1.b
; CHECK-NEXT:    cmpne p2.b, p1/z, z0.b, z1.b
; CHECK-NEXT:    and p0.b, p0/z, p0.b, p2.b
; CHECK-NEXT:    brkb p0.b, p1/z, p0.b
; CHECK-NEXT:    cntp x0, p0, p0.b
; CHECK-NEXT:    // kill: def $w0 killed $w0 killed $x0
; CHECK-NEXT:    ret
  %cmp = icmp ne <vscale x 16 x i8> %a, %b
  %select = select <vscale x 16 x i1> %pg, <vscale x 16 x i1> %cmp, <vscale x 16 x i1> zeroinitializer
  %and = and <vscale x 16 x i1> %pg, %select
  %res = call i32 @llvm.experimental.cttz.elts.i32.nxv16i1(<vscale x 16 x i1> %and, i1 0)
  ret i32 %res
}

define i64 @add_i64_ctz_nxv16i1_poison(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a, i64 %b) {
; CHECK-LABEL: add_i64_ctz_nxv16i1_poison:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.b
; CHECK-NEXT:    brkb p0.b, p0/z, p1.b
; CHECK-NEXT:    incp x0, p0.b
; CHECK-NEXT:    ret
  %res = call i64 @llvm.experimental.cttz.elts.i64.nxv16i1(<vscale x 16 x i1> %a, i1 1)
  %add = add i64 %res, %b
  ret i64 %add
}

define i32 @add_i32_ctz_nxv16i1_poison(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a, i32 %b) {
; CHECK-LABEL: add_i32_ctz_nxv16i1_poison:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.b
; CHECK-NEXT:    // kill: def $w0 killed $w0 def $x0
; CHECK-NEXT:    brkb p0.b, p0/z, p1.b
; CHECK-NEXT:    incp x0, p0.b
; CHECK-NEXT:    // kill: def $w0 killed $w0 killed $x0
; CHECK-NEXT:    ret
  %res = call i64 @llvm.experimental.cttz.elts.i64.nxv16i1(<vscale x 16 x i1> %a, i1 1)
  %trunc = trunc i64 %res to i32
  %add = add i32 %trunc, %b
  ret i32 %add
}

declare i32 @llvm.experimental.cttz.elts.i32.nxv8i1(<vscale x 8 x i1>, i1)
declare i64 @llvm.experimental.cttz.elts.i64.nxv8i1(<vscale x 8 x i1>, i1)
declare i64 @llvm.experimental.cttz.elts.i64.nxv16i1(<vscale x 16 x i1>, i1)
declare i32 @llvm.experimental.cttz.elts.i32.nxv16i1(<vscale x 16 x i1>, i1)
declare i32 @llvm.experimental.cttz.elts.i32.nxv32i1(<vscale x 32 x i1>, i1)
declare i32 @llvm.experimental.cttz.elts.i32.nxv4i32(<vscale x 4 x i32>, i1)

declare i64 @llvm.experimental.cttz.elts.i64.nxv16i8(<vscale x 16 x i8>, i1)

attributes #0 = { vscale_range(1,16) }
attributes #1 = { vscale_range(1,4096) }
