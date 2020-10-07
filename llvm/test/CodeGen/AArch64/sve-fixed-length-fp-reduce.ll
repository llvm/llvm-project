; RUN: llc -aarch64-sve-vector-bits-min=128  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=16 -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=32 -check-prefixes=CHECK,VBITS_EQ_256
; RUN: llc -aarch64-sve-vector-bits-min=384  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=32 -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=512  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=64 -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=640  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=64 -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=768  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=64 -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=896  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=64 -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1024 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1152 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1280 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1408 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1536 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1664 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1792 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1920 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=2048 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=256 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_GE_2048

target triple = "aarch64-unknown-linux-gnu"

; Don't use SVE when its registers are no bigger than NEON.
; NO_SVE-NOT: ptrue

;
; FMAXV
;

; No NEON 16-bit vector FMAXNMV support. Use SVE.
define half @fmaxv_v4f16(<4 x half> %a) #0 {
; CHECK-LABEL: fmaxv_v4f16:
; CHECK: fmaxnmv h0, v0.4h
; CHECK-NEXT: ret
  %res = call half @llvm.vector.reduce.fmax.v4f16(<4 x half> %a)
  ret half %res
}

; No NEON 16-bit vector FMAXNMV support. Use SVE.
define half @fmaxv_v8f16(<8 x half> %a) #0 {
; CHECK-LABEL: fmaxv_v8f16:
; CHECK: fmaxnmv h0, v0.8h
; CHECK-NEXT: ret
  %res = call half @llvm.vector.reduce.fmax.v8f16(<8 x half> %a)
  ret half %res
}

define half @fmaxv_v16f16(<16 x half>* %a) #0 {
; CHECK-LABEL: fmaxv_v16f16:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_GE_256-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_256-NEXT: fmaxnmv h0, [[PG]], [[OP]].h
; VBITS_GE_256-NEXT: ret
  %op = load <16 x half>, <16 x half>* %a
  %res = call half @llvm.vector.reduce.fmax.v16f16(<16 x half> %op)
  ret half %res
}

define half @fmaxv_v32f16(<32 x half>* %a) #0 {
; CHECK-LABEL: fmaxv_v32f16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: fmaxnmv h0, [[PG]], [[OP]].h
; VBITS_GE_512-NEXT: ret
  %op = load <32 x half>, <32 x half>* %a
  %res = call half @llvm.vector.reduce.fmax.v32f16(<32 x half> %op)
  ret half %res
}

define half @fmaxv_v64f16(<64 x half>* %a) #0 {
; CHECK-LABEL: fmaxv_v64f16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: fmaxnmv h0, [[PG]], [[OP]].h
; VBITS_GE_1024-NEXT: ret
  %op = load <64 x half>, <64 x half>* %a
  %res = call half @llvm.vector.reduce.fmax.v64f16(<64 x half> %op)
  ret half %res
}

define half @fmaxv_v128f16(<128 x half>* %a) #0 {
; CHECK-LABEL: fmaxv_v128f16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: fmaxnmv h0, [[PG]], [[OP]].h
; VBITS_GE_2048-NEXT: ret
  %op = load <128 x half>, <128 x half>* %a
  %res = call half @llvm.vector.reduce.fmax.v128f16(<128 x half> %op)
  ret half %res
}

; Don't use SVE for 64-bit f32 vectors.
define float @fmaxv_v2f32(<2 x float> %a) #0 {
; CHECK-LABEL: fmaxv_v2f32:
; CHECK: fmaxnmp s0, v0.2s
; CHECK: ret
  %res = call float @llvm.vector.reduce.fmax.v2f32(<2 x float> %a)
  ret float %res
}

; Don't use SVE for 128-bit f32 vectors.
define float @fmaxv_v4f32(<4 x float> %a) #0 {
; CHECK-LABEL: fmaxv_v4f32:
; CHECK: fmaxnmv s0, v0.4s
; CHECK: ret
  %res = call float @llvm.vector.reduce.fmax.v4f32(<4 x float> %a)
  ret float %res
}

define float @fmaxv_v8f32(<8 x float>* %a) #0 {
; CHECK-LABEL: fmaxv_v8f32:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_GE_256-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_256-NEXT: fmaxnmv s0, [[PG]], [[OP]].s
; VBITS_GE_256-NEXT: ret
  %op = load <8 x float>, <8 x float>* %a
  %res = call float @llvm.vector.reduce.fmax.v8f32(<8 x float> %op)
  ret float %res
}

define float @fmaxv_v16f32(<16 x float>* %a) #0 {
; CHECK-LABEL: fmaxv_v16f32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: fmaxnmv s0, [[PG]], [[OP]].s
; VBITS_GE_512-NEXT: ret
  %op = load <16 x float>, <16 x float>* %a
  %res = call float @llvm.vector.reduce.fmax.v16f32(<16 x float> %op)
  ret float %res
}

define float @fmaxv_v32f32(<32 x float>* %a) #0 {
; CHECK-LABEL: fmaxv_v32f32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: fmaxnmv s0, [[PG]], [[OP]].s
; VBITS_GE_1024-NEXT: ret
  %op = load <32 x float>, <32 x float>* %a
  %res = call float @llvm.vector.reduce.fmax.v32f32(<32 x float> %op)
  ret float %res
}

define float @fmaxv_v64f32(<64 x float>* %a) #0 {
; CHECK-LABEL: fmaxv_v64f32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: fmaxnmv s0, [[PG]], [[OP]].s
; VBITS_GE_2048-NEXT: ret
  %op = load <64 x float>, <64 x float>* %a
  %res = call float @llvm.vector.reduce.fmax.v64f32(<64 x float> %op)
  ret float %res
}

; Nothing to do for single element vectors.
define double @fmaxv_v1f64(<1 x double> %a) #0 {
; CHECK-LABEL: fmaxv_v1f64:
; CHECK-NOT: fmax
; CHECK: ret
  %res = call double @llvm.vector.reduce.fmax.v1f64(<1 x double> %a)
  ret double %res
}

; Don't use SVE for 128-bit f64 vectors.
define double @fmaxv_v2f64(<2 x double> %a) #0 {
; CHECK-LABEL: fmaxv_v2f64:
; CHECK: fmaxnmp d0, v0.2d
; CHECK-NEXT: ret
  %res = call double @llvm.vector.reduce.fmax.v2f64(<2 x double> %a)
  ret double %res
}

define double @fmaxv_v4f64(<4 x double>* %a) #0 {
; CHECK-LABEL: fmaxv_v4f64:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_GE_256-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_256-NEXT: fmaxnmv d0, [[PG]], [[OP]].d
; VBITS_GE_256-NEXT: ret
  %op = load <4 x double>, <4 x double>* %a
  %res = call double @llvm.vector.reduce.fmax.v4f64(<4 x double> %op)
  ret double %res
}

define double @fmaxv_v8f64(<8 x double>* %a) #0 {
; CHECK-LABEL: fmaxv_v8f64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: fmaxnmv d0, [[PG]], [[OP]].d
; VBITS_GE_512-NEXT: ret
  %op = load <8 x double>, <8 x double>* %a
  %res = call double @llvm.vector.reduce.fmax.v8f64(<8 x double> %op)
  ret double %res
}

define double @fmaxv_v16f64(<16 x double>* %a) #0 {
; CHECK-LABEL: fmaxv_v16f64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: fmaxnmv d0, [[PG]], [[OP]].d
; VBITS_GE_1024-NEXT: ret
  %op = load <16 x double>, <16 x double>* %a
  %res = call double @llvm.vector.reduce.fmax.v16f64(<16 x double> %op)
  ret double %res
}

define double @fmaxv_v32f64(<32 x double>* %a) #0 {
; CHECK-LABEL: fmaxv_v32f64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: fmaxnmv d0, [[PG]], [[OP]].d
; VBITS_GE_2048-NEXT: ret
  %op = load <32 x double>, <32 x double>* %a
  %res = call double @llvm.vector.reduce.fmax.v32f64(<32 x double> %op)
  ret double %res
}

;
; FMINV
;

; No NEON 16-bit vector FMINNMV support. Use SVE.
define half @fminv_v4f16(<4 x half> %a) #0 {
; CHECK-LABEL: fminv_v4f16:
; CHECK: fminnmv h0, v0.4h
; CHECK-NEXT: ret
  %res = call half @llvm.vector.reduce.fmin.v4f16(<4 x half> %a)
  ret half %res
}

; No NEON 16-bit vector FMINNMV support. Use SVE.
define half @fminv_v8f16(<8 x half> %a) #0 {
; CHECK-LABEL: fminv_v8f16:
; CHECK: fminnmv h0, v0.8h
; CHECK-NEXT: ret
  %res = call half @llvm.vector.reduce.fmin.v8f16(<8 x half> %a)
  ret half %res
}

define half @fminv_v16f16(<16 x half>* %a) #0 {
; CHECK-LABEL: fminv_v16f16:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_GE_256-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_256-NEXT: fminnmv h0, [[PG]], [[OP]].h
; VBITS_GE_256-NEXT: ret
  %op = load <16 x half>, <16 x half>* %a
  %res = call half @llvm.vector.reduce.fmin.v16f16(<16 x half> %op)
  ret half %res
}

define half @fminv_v32f16(<32 x half>* %a) #0 {
; CHECK-LABEL: fminv_v32f16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: fminnmv h0, [[PG]], [[OP]].h
; VBITS_GE_512-NEXT: ret
  %op = load <32 x half>, <32 x half>* %a
  %res = call half @llvm.vector.reduce.fmin.v32f16(<32 x half> %op)
  ret half %res
}

define half @fminv_v64f16(<64 x half>* %a) #0 {
; CHECK-LABEL: fminv_v64f16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: fminnmv h0, [[PG]], [[OP]].h
; VBITS_GE_1024-NEXT: ret
  %op = load <64 x half>, <64 x half>* %a
  %res = call half @llvm.vector.reduce.fmin.v64f16(<64 x half> %op)
  ret half %res
}

define half @fminv_v128f16(<128 x half>* %a) #0 {
; CHECK-LABEL: fminv_v128f16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: fminnmv h0, [[PG]], [[OP]].h
; VBITS_GE_2048-NEXT: ret
  %op = load <128 x half>, <128 x half>* %a
  %res = call half @llvm.vector.reduce.fmin.v128f16(<128 x half> %op)
  ret half %res
}

; Don't use SVE for 64-bit f32 vectors.
define float @fminv_v2f32(<2 x float> %a) #0 {
; CHECK-LABEL: fminv_v2f32:
; CHECK: fminnmp s0, v0.2s
; CHECK: ret
  %res = call float @llvm.vector.reduce.fmin.v2f32(<2 x float> %a)
  ret float %res
}

; Don't use SVE for 128-bit f32 vectors.
define float @fminv_v4f32(<4 x float> %a) #0 {
; CHECK-LABEL: fminv_v4f32:
; CHECK: fminnmv s0, v0.4s
; CHECK: ret
  %res = call float @llvm.vector.reduce.fmin.v4f32(<4 x float> %a)
  ret float %res
}

define float @fminv_v8f32(<8 x float>* %a) #0 {
; CHECK-LABEL: fminv_v8f32:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_GE_256-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_256-NEXT: fminnmv s0, [[PG]], [[OP]].s
; VBITS_GE_256-NEXT: ret
  %op = load <8 x float>, <8 x float>* %a
  %res = call float @llvm.vector.reduce.fmin.v8f32(<8 x float> %op)
  ret float %res
}

define float @fminv_v16f32(<16 x float>* %a) #0 {
; CHECK-LABEL: fminv_v16f32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: fminnmv s0, [[PG]], [[OP]].s
; VBITS_GE_512-NEXT: ret
  %op = load <16 x float>, <16 x float>* %a
  %res = call float @llvm.vector.reduce.fmin.v16f32(<16 x float> %op)
  ret float %res
}

define float @fminv_v32f32(<32 x float>* %a) #0 {
; CHECK-LABEL: fminv_v32f32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: fminnmv s0, [[PG]], [[OP]].s
; VBITS_GE_1024-NEXT: ret
  %op = load <32 x float>, <32 x float>* %a
  %res = call float @llvm.vector.reduce.fmin.v32f32(<32 x float> %op)
  ret float %res
}

define float @fminv_v64f32(<64 x float>* %a) #0 {
; CHECK-LABEL: fminv_v64f32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: fminnmv s0, [[PG]], [[OP]].s
; VBITS_GE_2048-NEXT: ret
  %op = load <64 x float>, <64 x float>* %a
  %res = call float @llvm.vector.reduce.fmin.v64f32(<64 x float> %op)
  ret float %res
}

; Nothing to do for single element vectors.
define double @fminv_v1f64(<1 x double> %a) #0 {
; CHECK-LABEL: fminv_v1f64:
; CHECK-NOT: fmin
; CHECK: ret
  %res = call double @llvm.vector.reduce.fmin.v1f64(<1 x double> %a)
  ret double %res
}

; Don't use SVE for 128-bit f64 vectors.
define double @fminv_v2f64(<2 x double> %a) #0 {
; CHECK-LABEL: fminv_v2f64:
; CHECK: fminnmp d0, v0.2d
; CHECK-NEXT: ret
  %res = call double @llvm.vector.reduce.fmin.v2f64(<2 x double> %a)
  ret double %res
}

define double @fminv_v4f64(<4 x double>* %a) #0 {
; CHECK-LABEL: fminv_v4f64:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_GE_256-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_256-NEXT: fminnmv d0, [[PG]], [[OP]].d
; VBITS_GE_256-NEXT: ret
  %op = load <4 x double>, <4 x double>* %a
  %res = call double @llvm.vector.reduce.fmin.v4f64(<4 x double> %op)
  ret double %res
}

define double @fminv_v8f64(<8 x double>* %a) #0 {
; CHECK-LABEL: fminv_v8f64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: fminnmv d0, [[PG]], [[OP]].d
; VBITS_GE_512-NEXT: ret
  %op = load <8 x double>, <8 x double>* %a
  %res = call double @llvm.vector.reduce.fmin.v8f64(<8 x double> %op)
  ret double %res
}

define double @fminv_v16f64(<16 x double>* %a) #0 {
; CHECK-LABEL: fminv_v16f64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: fminnmv d0, [[PG]], [[OP]].d
; VBITS_GE_1024-NEXT: ret
  %op = load <16 x double>, <16 x double>* %a
  %res = call double @llvm.vector.reduce.fmin.v16f64(<16 x double> %op)
  ret double %res
}

define double @fminv_v32f64(<32 x double>* %a) #0 {
; CHECK-LABEL: fminv_v32f64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: fminnmv d0, [[PG]], [[OP]].d
; VBITS_GE_2048-NEXT: ret
  %op = load <32 x double>, <32 x double>* %a
  %res = call double @llvm.vector.reduce.fmin.v32f64(<32 x double> %op)
  ret double %res
}

attributes #0 = { "target-features"="+sve" }

declare half @llvm.vector.reduce.fmax.v4f16(<4 x half>)
declare half @llvm.vector.reduce.fmax.v8f16(<8 x half>)
declare half @llvm.vector.reduce.fmax.v16f16(<16 x half>)
declare half @llvm.vector.reduce.fmax.v32f16(<32 x half>)
declare half @llvm.vector.reduce.fmax.v64f16(<64 x half>)
declare half @llvm.vector.reduce.fmax.v128f16(<128 x half>)

declare float @llvm.vector.reduce.fmax.v2f32(<2 x float>)
declare float @llvm.vector.reduce.fmax.v4f32(<4 x float>)
declare float @llvm.vector.reduce.fmax.v8f32(<8 x float>)
declare float @llvm.vector.reduce.fmax.v16f32(<16 x float>)
declare float @llvm.vector.reduce.fmax.v32f32(<32 x float>)
declare float @llvm.vector.reduce.fmax.v64f32(<64 x float>)

declare double @llvm.vector.reduce.fmax.v1f64(<1 x double>)
declare double @llvm.vector.reduce.fmax.v2f64(<2 x double>)
declare double @llvm.vector.reduce.fmax.v4f64(<4 x double>)
declare double @llvm.vector.reduce.fmax.v8f64(<8 x double>)
declare double @llvm.vector.reduce.fmax.v16f64(<16 x double>)
declare double @llvm.vector.reduce.fmax.v32f64(<32 x double>)

declare half @llvm.vector.reduce.fmin.v4f16(<4 x half>)
declare half @llvm.vector.reduce.fmin.v8f16(<8 x half>)
declare half @llvm.vector.reduce.fmin.v16f16(<16 x half>)
declare half @llvm.vector.reduce.fmin.v32f16(<32 x half>)
declare half @llvm.vector.reduce.fmin.v64f16(<64 x half>)
declare half @llvm.vector.reduce.fmin.v128f16(<128 x half>)

declare float @llvm.vector.reduce.fmin.v2f32(<2 x float>)
declare float @llvm.vector.reduce.fmin.v4f32(<4 x float>)
declare float @llvm.vector.reduce.fmin.v8f32(<8 x float>)
declare float @llvm.vector.reduce.fmin.v16f32(<16 x float>)
declare float @llvm.vector.reduce.fmin.v32f32(<32 x float>)
declare float @llvm.vector.reduce.fmin.v64f32(<64 x float>)

declare double @llvm.vector.reduce.fmin.v1f64(<1 x double>)
declare double @llvm.vector.reduce.fmin.v2f64(<2 x double>)
declare double @llvm.vector.reduce.fmin.v4f64(<4 x double>)
declare double @llvm.vector.reduce.fmin.v8f64(<8 x double>)
declare double @llvm.vector.reduce.fmin.v16f64(<16 x double>)
declare double @llvm.vector.reduce.fmin.v32f64(<32 x double>)
