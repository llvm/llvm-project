// RUN: llvm-mc -triple aarch64_lfi --aarch64-lfi-no-guard-elim %s | FileCheck %s
// LD1/ST1 single structure (no post-index)
ld1 { v0.b }[0], [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1 { v0.b }[0], [x28]
ld1 { v0.h }[1], [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1 { v0.h }[1], [x28]
ld1 { v0.s }[2], [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1 { v0.s }[2], [x28]
ld1 { v0.d }[1], [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1 { v0.d }[1], [x28]
st1 { v0.b }[0], [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st1 { v0.b }[0], [x28]
st1 { v0.h }[1], [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st1 { v0.h }[1], [x28]
st1 { v0.s }[2], [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st1 { v0.s }[2], [x28]
st1 { v0.d }[1], [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st1 { v0.d }[1], [x28]
// LD1/ST1 single structure with post-index (natural offset)
ld1 { v0.b }[0], [x0], #1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1 { v0.b }[0], [x28]
// CHECK-NEXT: add x0, x0, #1
ld1 { v0.h }[1], [x0], #2
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1 { v0.h }[1], [x28]
// CHECK-NEXT: add x0, x0, #2
ld1 { v0.s }[2], [x0], #4
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1 { v0.s }[2], [x28]
// CHECK-NEXT: add x0, x0, #4
ld1 { v0.d }[1], [x0], #8
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1 { v0.d }[1], [x28]
// CHECK-NEXT: add x0, x0, #8
st1 { v0.b }[0], [x0], #1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st1 { v0.b }[0], [x28]
// CHECK-NEXT: add x0, x0, #1
st1 { v0.h }[1], [x0], #2
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st1 { v0.h }[1], [x28]
// CHECK-NEXT: add x0, x0, #2
st1 { v0.s }[2], [x0], #4
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st1 { v0.s }[2], [x28]
// CHECK-NEXT: add x0, x0, #4
st1 { v0.d }[1], [x0], #8
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st1 { v0.d }[1], [x28]
// CHECK-NEXT: add x0, x0, #8
// LD1/ST1 single structure with post-index (register offset)
ld1 { v0.b }[0], [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1 { v0.b }[0], [x28]
// CHECK-NEXT: add x0, x0, x1
ld1 { v0.s }[2], [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1 { v0.s }[2], [x28]
// CHECK-NEXT: add x0, x0, x1
st1 { v0.d }[1], [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st1 { v0.d }[1], [x28]
// CHECK-NEXT: add x0, x0, x1
// LD1R (replicate single element to all lanes)
ld1r { v0.8b }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1r { v0.8b }, [x28]
ld1r { v0.16b }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1r { v0.16b }, [x28]
ld1r { v0.4h }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1r { v0.4h }, [x28]
ld1r { v0.8h }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1r { v0.8h }, [x28]
ld1r { v0.2s }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1r { v0.2s }, [x28]
ld1r { v0.4s }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1r { v0.4s }, [x28]
ld1r { v0.1d }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1r { v0.1d }, [x28]
ld1r { v0.2d }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1r { v0.2d }, [x28]
// LD1R with post-index
ld1r { v0.8b }, [x0], #1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1r { v0.8b }, [x28]
// CHECK-NEXT: add x0, x0, #1
ld1r { v0.4h }, [x0], #2
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1r { v0.4h }, [x28]
// CHECK-NEXT: add x0, x0, #2
ld1r { v0.2s }, [x0], #4
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1r { v0.2s }, [x28]
// CHECK-NEXT: add x0, x0, #4
ld1r { v0.1d }, [x0], #8
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1r { v0.1d }, [x28]
// CHECK-NEXT: add x0, x0, #8
ld1r { v0.2d }, [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1r { v0.2d }, [x28]
// CHECK-NEXT: add x0, x0, x1
// LD1/ST1 multiple structures (1-4 registers)
ld1 { v0.16b }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1 { v0.16b }, [x28]
ld1 { v0.16b, v1.16b }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1 { v0.16b, v1.16b }, [x28]
ld1 { v0.16b, v1.16b, v2.16b }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1 { v0.16b, v1.16b, v2.16b }, [x28]
ld1 { v0.16b, v1.16b, v2.16b, v3.16b }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1 { v0.16b, v1.16b, v2.16b, v3.16b }, [x28]
st1 { v0.16b }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st1 { v0.16b }, [x28]
st1 { v0.16b, v1.16b }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st1 { v0.16b, v1.16b }, [x28]
st1 { v0.16b, v1.16b, v2.16b }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st1 { v0.16b, v1.16b, v2.16b }, [x28]
st1 { v0.16b, v1.16b, v2.16b, v3.16b }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st1 { v0.16b, v1.16b, v2.16b, v3.16b }, [x28]
// LD1/ST1 multiple structures with post-index
ld1 { v0.16b }, [x0], #16
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1 { v0.16b }, [x28]
// CHECK-NEXT: add x0, x0, #16
ld1 { v0.16b, v1.16b }, [x0], #32
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1 { v0.16b, v1.16b }, [x28]
// CHECK-NEXT: add x0, x0, #32
ld1 { v0.16b, v1.16b, v2.16b }, [x0], #48
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1 { v0.16b, v1.16b, v2.16b }, [x28]
// CHECK-NEXT: add x0, x0, #48
ld1 { v0.16b, v1.16b, v2.16b, v3.16b }, [x0], #64
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1 { v0.16b, v1.16b, v2.16b, v3.16b }, [x28]
// CHECK-NEXT: add x0, x0, #64
ld1 { v0.16b }, [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld1 { v0.16b }, [x28]
// CHECK-NEXT: add x0, x0, x1
// LD2/ST2 multiple structures
ld2 { v0.8b, v1.8b }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld2 { v0.8b, v1.8b }, [x28]
ld2 { v0.16b, v1.16b }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld2 { v0.16b, v1.16b }, [x28]
ld2 { v0.4h, v1.4h }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld2 { v0.4h, v1.4h }, [x28]
ld2 { v0.8h, v1.8h }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld2 { v0.8h, v1.8h }, [x28]
ld2 { v0.2s, v1.2s }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld2 { v0.2s, v1.2s }, [x28]
ld2 { v0.4s, v1.4s }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld2 { v0.4s, v1.4s }, [x28]
ld2 { v0.2d, v1.2d }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld2 { v0.2d, v1.2d }, [x28]
st2 { v0.16b, v1.16b }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st2 { v0.16b, v1.16b }, [x28]
ld2 { v0.16b, v1.16b }, [x0], #32
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld2 { v0.16b, v1.16b }, [x28]
// CHECK-NEXT: add x0, x0, #32
st2 { v0.16b, v1.16b }, [x0], #32
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st2 { v0.16b, v1.16b }, [x28]
// CHECK-NEXT: add x0, x0, #32
// LD3/ST3 multiple structures
ld3 { v0.8b, v1.8b, v2.8b }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld3 { v0.8b, v1.8b, v2.8b }, [x28]
ld3 { v0.16b, v1.16b, v2.16b }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld3 { v0.16b, v1.16b, v2.16b }, [x28]
ld3 { v0.4s, v1.4s, v2.4s }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld3 { v0.4s, v1.4s, v2.4s }, [x28]
st3 { v0.4s, v1.4s, v2.4s }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st3 { v0.4s, v1.4s, v2.4s }, [x28]
ld3 { v0.4s, v1.4s, v2.4s }, [x0], #48
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld3 { v0.4s, v1.4s, v2.4s }, [x28]
// CHECK-NEXT: add x0, x0, #48
ld3 { v0.4s, v1.4s, v2.4s }, [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld3 { v0.4s, v1.4s, v2.4s }, [x28]
// CHECK-NEXT: add x0, x0, x1
// LD4/ST4 multiple structures
ld4 { v0.8b, v1.8b, v2.8b, v3.8b }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld4 { v0.8b, v1.8b, v2.8b, v3.8b }, [x28]
ld4 { v0.16b, v1.16b, v2.16b, v3.16b }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld4 { v0.16b, v1.16b, v2.16b, v3.16b }, [x28]
ld4 { v0.4s, v1.4s, v2.4s, v3.4s }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld4 { v0.4s, v1.4s, v2.4s, v3.4s }, [x28]
ld4 { v0.2d, v1.2d, v2.2d, v3.2d }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld4 { v0.2d, v1.2d, v2.2d, v3.2d }, [x28]
st4 { v0.4s, v1.4s, v2.4s, v3.4s }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st4 { v0.4s, v1.4s, v2.4s, v3.4s }, [x28]
ld4 { v0.4s, v1.4s, v2.4s, v3.4s }, [x0], #64
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld4 { v0.4s, v1.4s, v2.4s, v3.4s }, [x28]
// CHECK-NEXT: add x0, x0, #64
ld4 { v0.4s, v1.4s, v2.4s, v3.4s }, [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld4 { v0.4s, v1.4s, v2.4s, v3.4s }, [x28]
// CHECK-NEXT: add x0, x0, x1
// LD2R/LD3R/LD4R (replicate)
ld2r { v0.8b, v1.8b }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld2r { v0.8b, v1.8b }, [x28]
ld3r { v0.4s, v1.4s, v2.4s }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld3r { v0.4s, v1.4s, v2.4s }, [x28]
ld4r { v0.2d, v1.2d, v2.2d, v3.2d }, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld4r { v0.2d, v1.2d, v2.2d, v3.2d }, [x28]
ld2r { v0.8b, v1.8b }, [x0], #2
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld2r { v0.8b, v1.8b }, [x28]
// CHECK-NEXT: add x0, x0, #2
ld3r { v0.4s, v1.4s, v2.4s }, [x0], #12
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld3r { v0.4s, v1.4s, v2.4s }, [x28]
// CHECK-NEXT: add x0, x0, #12
ld4r { v0.2d, v1.2d, v2.2d, v3.2d }, [x0], #32
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld4r { v0.2d, v1.2d, v2.2d, v3.2d }, [x28]
// CHECK-NEXT: add x0, x0, #32
// LD2/LD3/LD4 single structure (lane loads)
ld2 { v0.b, v1.b }[0], [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld2 { v0.b, v1.b }[0], [x28]
ld3 { v0.s, v1.s, v2.s }[1], [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld3 { v0.s, v1.s, v2.s }[1], [x28]
ld4 { v0.d, v1.d, v2.d, v3.d }[0], [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld4 { v0.d, v1.d, v2.d, v3.d }[0], [x28]
st2 { v0.h, v1.h }[3], [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st2 { v0.h, v1.h }[3], [x28]
st3 { v0.s, v1.s, v2.s }[2], [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st3 { v0.s, v1.s, v2.s }[2], [x28]
st4 { v0.d, v1.d, v2.d, v3.d }[1], [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st4 { v0.d, v1.d, v2.d, v3.d }[1], [x28]

ld2 { v0.b, v1.b }[0], [x0], #2
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld2 { v0.b, v1.b }[0], [x28]
// CHECK-NEXT: add x0, x0, #2
ld3 { v0.s, v1.s, v2.s }[1], [x0], #12
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld3 { v0.s, v1.s, v2.s }[1], [x28]
// CHECK-NEXT: add x0, x0, #12
ld4 { v0.d, v1.d, v2.d, v3.d }[0], [x0], #32
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld4 { v0.d, v1.d, v2.d, v3.d }[0], [x28]
// CHECK-NEXT: add x0, x0, #32
ld2 { v0.s, v1.s }[1], [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld2 { v0.s, v1.s }[1], [x28]
// CHECK-NEXT: add x0, x0, x1

// ST2/ST3/ST4 lane stores with immediate post-index
st2 { v0.b, v1.b }[0], [x0], #2
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st2 { v0.b, v1.b }[0], [x28]
// CHECK-NEXT: add x0, x0, #2
st2 { v0.h, v1.h }[1], [x0], #4
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st2 { v0.h, v1.h }[1], [x28]
// CHECK-NEXT: add x0, x0, #4
st2 { v0.s, v1.s }[2], [x0], #8
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st2 { v0.s, v1.s }[2], [x28]
// CHECK-NEXT: add x0, x0, #8
st2 { v0.d, v1.d }[1], [x0], #16
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st2 { v0.d, v1.d }[1], [x28]
// CHECK-NEXT: add x0, x0, #16
st3 { v0.b, v1.b, v2.b }[0], [x0], #3
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st3 { v0.b, v1.b, v2.b }[0], [x28]
// CHECK-NEXT: add x0, x0, #3
st3 { v0.h, v1.h, v2.h }[1], [x0], #6
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st3 { v0.h, v1.h, v2.h }[1], [x28]
// CHECK-NEXT: add x0, x0, #6
st3 { v0.s, v1.s, v2.s }[2], [x0], #12
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st3 { v0.s, v1.s, v2.s }[2], [x28]
// CHECK-NEXT: add x0, x0, #12
st3 { v0.d, v1.d, v2.d }[0], [x0], #24
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st3 { v0.d, v1.d, v2.d }[0], [x28]
// CHECK-NEXT: add x0, x0, #24
st4 { v0.b, v1.b, v2.b, v3.b }[0], [x0], #4
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st4 { v0.b, v1.b, v2.b, v3.b }[0], [x28]
// CHECK-NEXT: add x0, x0, #4
st4 { v0.h, v1.h, v2.h, v3.h }[1], [x0], #8
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st4 { v0.h, v1.h, v2.h, v3.h }[1], [x28]
// CHECK-NEXT: add x0, x0, #8
st4 { v0.s, v1.s, v2.s, v3.s }[2], [x0], #16
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st4 { v0.s, v1.s, v2.s, v3.s }[2], [x28]
// CHECK-NEXT: add x0, x0, #16
st4 { v0.d, v1.d, v2.d, v3.d }[0], [x0], #32
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st4 { v0.d, v1.d, v2.d, v3.d }[0], [x28]
// CHECK-NEXT: add x0, x0, #32

// ST1/ST2/ST3/ST4 lane stores with register post-index
st1 { v0.b }[0], [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st1 { v0.b }[0], [x28]
// CHECK-NEXT: add x0, x0, x1
st1 { v0.h }[1], [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st1 { v0.h }[1], [x28]
// CHECK-NEXT: add x0, x0, x1
st1 { v0.s }[2], [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st1 { v0.s }[2], [x28]
// CHECK-NEXT: add x0, x0, x1
st2 { v0.b, v1.b }[0], [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st2 { v0.b, v1.b }[0], [x28]
// CHECK-NEXT: add x0, x0, x1
st2 { v0.s, v1.s }[1], [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st2 { v0.s, v1.s }[1], [x28]
// CHECK-NEXT: add x0, x0, x1
st2 { v0.d, v1.d }[0], [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st2 { v0.d, v1.d }[0], [x28]
// CHECK-NEXT: add x0, x0, x1
st3 { v0.b, v1.b, v2.b }[0], [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st3 { v0.b, v1.b, v2.b }[0], [x28]
// CHECK-NEXT: add x0, x0, x1
st3 { v0.s, v1.s, v2.s }[1], [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st3 { v0.s, v1.s, v2.s }[1], [x28]
// CHECK-NEXT: add x0, x0, x1
st3 { v0.d, v1.d, v2.d }[0], [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st3 { v0.d, v1.d, v2.d }[0], [x28]
// CHECK-NEXT: add x0, x0, x1
st4 { v0.b, v1.b, v2.b, v3.b }[0], [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st4 { v0.b, v1.b, v2.b, v3.b }[0], [x28]
// CHECK-NEXT: add x0, x0, x1
st4 { v0.s, v1.s, v2.s, v3.s }[1], [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st4 { v0.s, v1.s, v2.s, v3.s }[1], [x28]
// CHECK-NEXT: add x0, x0, x1
st4 { v0.d, v1.d, v2.d, v3.d }[0], [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st4 { v0.d, v1.d, v2.d, v3.d }[0], [x28]
// CHECK-NEXT: add x0, x0, x1

ld3 { v0.b, v1.b, v2.b }[0], [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld3 { v0.b, v1.b, v2.b }[0], [x28]
// CHECK-NEXT: add x0, x0, x1
ld3 { v0.d, v1.d, v2.d }[0], [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld3 { v0.d, v1.d, v2.d }[0], [x28]
// CHECK-NEXT: add x0, x0, x1
ld4 { v0.b, v1.b, v2.b, v3.b }[0], [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld4 { v0.b, v1.b, v2.b, v3.b }[0], [x28]
// CHECK-NEXT: add x0, x0, x1
ld4 { v0.s, v1.s, v2.s, v3.s }[1], [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld4 { v0.s, v1.s, v2.s, v3.s }[1], [x28]
// CHECK-NEXT: add x0, x0, x1

// ST1/ST2/ST3/ST4 multi-register stores with register post-index
st1 { v0.16b }, [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st1 { v0.16b }, [x28]
// CHECK-NEXT: add x0, x0, x1
st1 { v0.16b, v1.16b }, [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st1 { v0.16b, v1.16b }, [x28]
// CHECK-NEXT: add x0, x0, x1
st1 { v0.16b, v1.16b, v2.16b }, [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st1 { v0.16b, v1.16b, v2.16b }, [x28]
// CHECK-NEXT: add x0, x0, x1
st1 { v0.16b, v1.16b, v2.16b, v3.16b }, [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st1 { v0.16b, v1.16b, v2.16b, v3.16b }, [x28]
// CHECK-NEXT: add x0, x0, x1
st2 { v0.16b, v1.16b }, [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st2 { v0.16b, v1.16b }, [x28]
// CHECK-NEXT: add x0, x0, x1
st2 { v0.4s, v1.4s }, [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st2 { v0.4s, v1.4s }, [x28]
// CHECK-NEXT: add x0, x0, x1
st3 { v0.16b, v1.16b, v2.16b }, [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st3 { v0.16b, v1.16b, v2.16b }, [x28]
// CHECK-NEXT: add x0, x0, x1
st3 { v0.4s, v1.4s, v2.4s }, [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st3 { v0.4s, v1.4s, v2.4s }, [x28]
// CHECK-NEXT: add x0, x0, x1
st4 { v0.16b, v1.16b, v2.16b, v3.16b }, [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st4 { v0.16b, v1.16b, v2.16b, v3.16b }, [x28]
// CHECK-NEXT: add x0, x0, x1
st4 { v0.4s, v1.4s, v2.4s, v3.4s }, [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st4 { v0.4s, v1.4s, v2.4s, v3.4s }, [x28]
// CHECK-NEXT: add x0, x0, x1
st4 { v0.2d, v1.2d, v2.2d, v3.2d }, [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: st4 { v0.2d, v1.2d, v2.2d, v3.2d }, [x28]
// CHECK-NEXT: add x0, x0, x1

ld2 { v0.16b, v1.16b }, [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld2 { v0.16b, v1.16b }, [x28]
// CHECK-NEXT: add x0, x0, x1
ld2 { v0.4s, v1.4s }, [x0], x1
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ld2 { v0.4s, v1.4s }, [x28]
// CHECK-NEXT: add x0, x0, x1
