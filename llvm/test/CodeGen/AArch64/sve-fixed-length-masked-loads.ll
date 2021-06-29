; RUN: llc -aarch64-sve-vector-bits-min=128  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=16  -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=32  -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=384  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=32  -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=512  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=640  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=768  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=896  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1024 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1152 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1280 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1408 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1536 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1664 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1792 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1920 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=2048 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=256 -check-prefixes=CHECK,VBITS_GE_2048,VBITS_GE_1024,VBITS_GE_512

target triple = "aarch64-unknown-linux-gnu"

; Don't use SVE when its registers are no bigger than NEON.
; NO_SVE-NOT: ptrue

;
; Masked Loads
;
define <2 x half> @masked_load_v2f16(<2 x half>* %ap, <2 x half>* %bp) #0 {
; CHECK-LABEL: masked_load_v2f16:
; CHECK: ldr s[[N0:[0-9]+]], [x0]
; CHECK-NEXT: ldr s[[N1:[0-9]+]], [x1]
; CHECK-NEXT: ptrue [[PG0:p[0-9]+]].h, vl4
; CHECK-NEXT: fcmeq v[[N2:[0-9]+]].4h, v[[N0]].4h, v[[N1]].4h
; CHECK-NEXT: umov [[W0:w[0-9]+]], v[[N2]].h[0]
; CHECK-NEXT: umov [[W1:w[0-9]+]], v[[N2]].h[1]
; CHECK-NEXT: fmov s[[V0:[0-9]+]], [[W0]]
; CHECK-NEXT: mov v[[V0]].s[1], [[W1]]
; CHECK-NEXT: shl v[[V0]].2s, v[[V0]].2s, #16
; CHECK-NEXT: sshr v[[V0]].2s, v[[V0]].2s, #16
; CHECK-NEXT: movi [[D0:d[0-9]+]], #0000000000000000
; CHECK-NEXT: fmov [[W1]], s[[V0]]
; CHECK-NEXT: mov [[W0]], v[[V0]].s[1]
; CHECK-NEXT: mov [[V1:v[0-9]+]].h[0], [[W1]]
; CHECK-NEXT: mov [[V1]].h[1], [[W0]]
; CHECK-NEXT: shl v[[V0]].4h, [[V1]].4h, #15
; CHECK-NEXT: sshr v[[V0]].4h, v[[V0]].4h, #15
; CHECK-NEXT: cmpne [[PG1:p[0-9]+]].h, [[PG0]]/z, z[[N2]].h, #0
; CHECK-NEXT: ld1h { z0.h }, [[PG1]]/z, [x0]
; CHECK-NEXT: ret
  %a = load <2 x half>, <2 x half>* %ap
  %b = load <2 x half>, <2 x half>* %bp
  %mask = fcmp oeq <2 x half> %a, %b
  %load = call <2 x half> @llvm.masked.load.v2f16(<2 x half>* %ap, i32 8, <2 x i1> %mask, <2 x half> zeroinitializer)
  ret <2 x half> %load
}

define <2 x float> @masked_load_v2f32(<2 x float>* %ap, <2 x float>* %bp) #0 {
; CHECK-LABEL: masked_load_v2f32:
; CHECK: ldr d[[N0:[0-9]+]], [x0]
; CHECK-NEXT: ldr d[[N1:[0-9]+]], [x1]
; CHECK-NEXT: ptrue [[PG0:p[0-9]+]].s, vl2
; CHECK-NEXT: fcmeq v[[N2:[0-9]+]].2s, v[[N0]].2s, v[[N1]].2s
; CHECK-NEXT: cmpne [[PG1:p[0-9]+]].s, [[PG0]]/z, z[[N2]].s, #0
; CHECK-NEXT: ld1w { z0.s }, [[PG1]]/z, [x0]
; CHECK-NEXT: ret
  %a = load <2 x float>, <2 x float>* %ap
  %b = load <2 x float>, <2 x float>* %bp
  %mask = fcmp oeq <2 x float> %a, %b
  %load = call <2 x float> @llvm.masked.load.v2f32(<2 x float>* %ap, i32 8, <2 x i1> %mask, <2 x float> zeroinitializer)
  ret <2 x float> %load
}

define <4 x float> @masked_load_v4f32(<4 x float>* %ap, <4 x float>* %bp) #0 {
; CHECK-LABEL: masked_load_v4f32:
; CHECK: ldr q[[N0:[0-9]+]], [x0]
; CHECK-NEXT: ldr q[[N1:[0-9]+]], [x1]
; CHECK-NEXT: ptrue [[PG0:p[0-9]+]].s, vl4
; CHECK-NEXT: fcmeq v[[N2:[0-9]+]].4s, v[[N0]].4s, v[[N1]].4s
; CHECK-NEXT: cmpne [[PG1:p[0-9]+]].s, [[PG0]]/z, z[[N2]].s, #0
; CHECK-NEXT: ld1w { z0.s }, [[PG1]]/z, [x0]
; CHECK-NEXT: ret
  %a = load <4 x float>, <4 x float>* %ap
  %b = load <4 x float>, <4 x float>* %bp
  %mask = fcmp oeq <4 x float> %a, %b
  %load = call <4 x float> @llvm.masked.load.v4f32(<4 x float>* %ap, i32 8, <4 x i1> %mask, <4 x float> zeroinitializer)
  ret <4 x float> %load
}

define <8 x float> @masked_load_v8f32(<8 x float>* %ap, <8 x float>* %bp) #0 {
; CHECK-LABEL: masked_load_v8f32:
; CHECK: ptrue [[PG0:p[0-9]+]].s, vl[[#min(div(VBYTES,4),8)]]
; CHECK-NEXT: ld1w { [[Z0:z[0-9]+]].s }, p0/z, [x0]
; CHECK-NEXT: ld1w { [[Z1:z[0-9]+]].s }, p0/z, [x1]
; CHECK-NEXT: fcmeq [[PG1:p[0-9]+]].s, [[PG0]]/z, [[Z0]].s, [[Z1]].s
; CHECK-NEXT: ld1w { [[Z0]].s }, [[PG1]]/z, [x0]
; CHECK-NEXT: st1w { [[Z0]].s }, [[PG0]], [x8]
; CHECK-NEXT: ret
  %a = load <8 x float>, <8 x float>* %ap
  %b = load <8 x float>, <8 x float>* %bp
  %mask = fcmp oeq <8 x float> %a, %b
  %load = call <8 x float> @llvm.masked.load.v8f32(<8 x float>* %ap, i32 8, <8 x i1> %mask, <8 x float> zeroinitializer)
  ret <8 x float> %load
}

define <16 x float> @masked_load_v16f32(<16 x float>* %ap, <16 x float>* %bp) #0 {
; CHECK-LABEL: masked_load_v16f32:
; VBITS_GE_512: ptrue [[PG0:p[0-9]+]].s, vl[[#min(div(VBYTES,4),16)]]
; VBITS_GE_512-NEXT: ld1w { [[Z0:z[0-9]+]].s }, p0/z, [x0]
; VBITS_GE_512-NEXT: ld1w { [[Z1:z[0-9]+]].s }, p0/z, [x1]
; VBITS_GE_512-NEXT: fcmeq [[PG1:p[0-9]+]].s, [[PG0]]/z, [[Z0]].s, [[Z1]].s
; VBITS_GE_512-NEXT: ld1w { [[Z0]].s }, [[PG1]]/z, [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: st1w { [[Z0]].s }, [[PG0]], [x8]
; VBITS_GE_512-NEXT: ret
  %a = load <16 x float>, <16 x float>* %ap
  %b = load <16 x float>, <16 x float>* %bp
  %mask = fcmp oeq <16 x float> %a, %b
  %load = call <16 x float> @llvm.masked.load.v16f32(<16 x float>* %ap, i32 8, <16 x i1> %mask, <16 x float> zeroinitializer)
  ret <16 x float> %load
}

define <32 x float> @masked_load_v32f32(<32 x float>* %ap, <32 x float>* %bp) #0 {
; CHECK-LABEL: masked_load_v32f32:
; VBITS_GE_1024: ptrue [[PG0:p[0-9]+]].s, vl[[#min(div(VBYTES,4),32)]]
; VBITS_GE_1024-NEXT: ld1w { [[Z0:z[0-9]+]].s }, p0/z, [x0]
; VBITS_GE_1024-NEXT: ld1w { [[Z1:z[0-9]+]].s }, p0/z, [x1]
; VBITS_GE_1024-NEXT: fcmeq [[PG1:p[0-9]+]].s, [[PG0]]/z, [[Z0]].s, [[Z1]].s
; VBITS_GE_1024-NEXT: ld1w { [[Z0]].s }, [[PG1]]/z, [x{{[0-9]+}}]
; VBITS_GE_1024-NEXT: st1w { [[Z0]].s }, [[PG0]], [x8]
; VBITS_GE_1024-NEXT: ret
  %a = load <32 x float>, <32 x float>* %ap
  %b = load <32 x float>, <32 x float>* %bp
  %mask = fcmp oeq <32 x float> %a, %b
  %load = call <32 x float> @llvm.masked.load.v32f32(<32 x float>* %ap, i32 8, <32 x i1> %mask, <32 x float> zeroinitializer)
  ret <32 x float> %load
}

define <64 x float> @masked_load_v64f32(<64 x float>* %ap, <64 x float>* %bp) #0 {
; CHECK-LABEL: masked_load_v64f32:
; VBITS_GE_2048: ptrue [[PG0:p[0-9]+]].s, vl[[#min(div(VBYTES,4),64)]]
; VBITS_GE_2048-NEXT: ld1w { [[Z0:z[0-9]+]].s }, p0/z, [x0]
; VBITS_GE_2048-NEXT: ld1w { [[Z1:z[0-9]+]].s }, p0/z, [x1]
; VBITS_GE_2048-NEXT: fcmeq [[PG1:p[0-9]+]].s, [[PG0]]/z, [[Z0]].s, [[Z1]].s
; VBITS_GE_2048-NEXT: ld1w { [[Z0]].s }, [[PG1]]/z, [x{{[0-9]+}}]
; VBITS_GE_2048-NEXT: st1w { [[Z0]].s }, [[PG0]], [x8]
; VBITS_GE_2048-NEXT: ret

  %a = load <64 x float>, <64 x float>* %ap
  %b = load <64 x float>, <64 x float>* %bp
  %mask = fcmp oeq <64 x float> %a, %b
  %load = call <64 x float> @llvm.masked.load.v64f32(<64 x float>* %ap, i32 8, <64 x i1> %mask, <64 x float> zeroinitializer)
  ret <64 x float> %load
}

define <64 x i8> @masked_load_v64i8(<64 x i8>* %ap, <64 x i8>* %bp) #0 {
; CHECK-LABEL: masked_load_v64i8:
; VBITS_GE_512: ptrue [[PG0:p[0-9]+]].b, vl64
; VBITS_GE_512-NEXT: ld1b { [[Z0:z[0-9]+]].b }, p0/z, [x0]
; VBITS_GE_512-NEXT: ld1b { [[Z1:z[0-9]+]].b }, p0/z, [x1]
; VBITS_GE_512-NEXT: cmpeq [[PG1:p[0-9]+]].b, [[PG0]]/z, [[Z0]].b, [[Z1]].b
; VBITS_GE_512-NEXT: ld1b { [[Z0]].b }, [[PG1]]/z, [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: st1b { [[Z0]].b }, [[PG0]], [x8]
; VBITS_GE_512-NEXT: ret
  %a = load <64 x i8>, <64 x i8>* %ap
  %b = load <64 x i8>, <64 x i8>* %bp
  %mask = icmp eq <64 x i8> %a, %b
  %load = call <64 x i8> @llvm.masked.load.v64i8(<64 x i8>* %ap, i32 8, <64 x i1> %mask, <64 x i8> undef)
  ret <64 x i8> %load
}

define <32 x i16> @masked_load_v32i16(<32 x i16>* %ap, <32 x i16>* %bp) #0 {
; CHECK-LABEL: masked_load_v32i16:
; VBITS_GE_512: ptrue [[PG0:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: ld1h { [[Z0:z[0-9]+]].h }, p0/z, [x0]
; VBITS_GE_512-NEXT: ld1h { [[Z1:z[0-9]+]].h }, p0/z, [x1]
; VBITS_GE_512-NEXT: cmpeq [[PG1:p[0-9]+]].h, [[PG0]]/z, [[Z0]].h, [[Z1]].h
; VBITS_GE_512-NEXT: ld1h { [[Z0]].h }, [[PG1]]/z, [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: st1h { [[Z0]].h }, [[PG0]], [x8]
; VBITS_GE_512: ret
  %a = load <32 x i16>, <32 x i16>* %ap
  %b = load <32 x i16>, <32 x i16>* %bp
  %mask = icmp eq <32 x i16> %a, %b
  %load = call <32 x i16> @llvm.masked.load.v32i16(<32 x i16>* %ap, i32 8, <32 x i1> %mask, <32 x i16> undef)
  ret <32 x i16> %load
}

define <16 x i32> @masked_load_v16i32(<16 x i32>* %ap, <16 x i32>* %bp) #0 {
; CHECK-LABEL: masked_load_v16i32:
; VBITS_GE_512: ptrue [[PG0:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[Z0:z[0-9]+]].s }, p0/z, [x0]
; VBITS_GE_512-NEXT: ld1w { [[Z1:z[0-9]+]].s }, p0/z, [x1]
; VBITS_GE_512-NEXT: cmpeq [[PG1:p[0-9]+]].s, [[PG0]]/z, [[Z0]].s, [[Z1]].s
; VBITS_GE_512-NEXT: ld1w { [[Z0]].s }, [[PG1]]/z, [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: st1w { [[Z0]].s }, [[PG0]], [x8]
; VBITS_GE_512-NEXT: ret
  %a = load <16 x i32>, <16 x i32>* %ap
  %b = load <16 x i32>, <16 x i32>* %bp
  %mask = icmp eq <16 x i32> %a, %b
  %load = call <16 x i32> @llvm.masked.load.v16i32(<16 x i32>* %ap, i32 8, <16 x i1> %mask, <16 x i32> undef)
  ret <16 x i32> %load
}

define <8 x i64> @masked_load_v8i64(<8 x i64>* %ap, <8 x i64>* %bp) #0 {
; CHECK-LABEL: masked_load_v8i64:
; VBITS_GE_512: ptrue [[PG0:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[Z0:z[0-9]+]].d }, p0/z, [x0]
; VBITS_GE_512-NEXT: ld1d { [[Z1:z[0-9]+]].d }, p0/z, [x1]
; VBITS_GE_512-NEXT: cmpeq [[PG1:p[0-9]+]].d, [[PG0]]/z, [[Z0]].d, [[Z1]].d
; VBITS_GE_512-NEXT: ld1d { [[Z0]].d }, [[PG1]]/z, [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: st1d { [[Z0]].d }, [[PG0]], [x8]
; VBITS_GE_512-NEXT: ret
  %a = load <8 x i64>, <8 x i64>* %ap
  %b = load <8 x i64>, <8 x i64>* %bp
  %mask = icmp eq <8 x i64> %a, %b
  %load = call <8 x i64> @llvm.masked.load.v8i64(<8 x i64>* %ap, i32 8, <8 x i1> %mask, <8 x i64> undef)
  ret <8 x i64> %load
}

define <8 x i64> @masked_load_passthru_v8i64(<8 x i64>* %ap, <8 x i64>* %bp) #0 {
; CHECK-LABEL: masked_load_passthru_v8i64:
; VBITS_GE_512: ptrue [[PG0:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[Z0:z[0-9]+]].d }, p0/z, [x0]
; VBITS_GE_512-NEXT: ld1d { [[Z1:z[0-9]+]].d }, p0/z, [x1]
; VBITS_GE_512-NEXT: cmpeq [[PG1:p[0-9]+]].d, [[PG0]]/z, [[Z0]].d, [[Z1]].d
; VBITS_GE_512-NEXT: ld1d { [[Z0]].d }, [[PG1]]/z, [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: sel [[Z2:z[0-9]+]].d, [[PG1]], [[Z0]].d, [[Z1]].d
; VBITS_GE_512-NEXT: st1d { [[Z2]].d }, [[PG0]], [x8]
; VBITS_GE_512-NEXT: ret
  %a = load <8 x i64>, <8 x i64>* %ap
  %b = load <8 x i64>, <8 x i64>* %bp
  %mask = icmp eq <8 x i64> %a, %b
  %load = call <8 x i64> @llvm.masked.load.v8i64(<8 x i64>* %ap, i32 8, <8 x i1> %mask, <8 x i64> %b)
  ret <8 x i64> %load
}

define <8 x double> @masked_load_passthru_v8f64(<8 x double>* %ap, <8 x double>* %bp) #0 {
; CHECK-LABEL: masked_load_passthru_v8f64:
; VBITS_GE_512: ptrue [[PG0:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[Z0:z[0-9]+]].d }, p0/z, [x0]
; VBITS_GE_512-NEXT: ld1d { [[Z1:z[0-9]+]].d }, p0/z, [x1]
; VBITS_GE_512-NEXT: fcmeq [[PG1:p[0-9]+]].d, [[PG0]]/z, [[Z0]].d, [[Z1]].d
; VBITS_GE_512-NEXT: ld1d { [[Z0]].d }, [[PG1]]/z, [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: sel [[Z2:z[0-9]+]].d, [[PG1]], [[Z0]].d, [[Z1]].d
; VBITS_GE_512-NEXT: st1d { [[Z2]].d }, [[PG0]], [x8]
; VBITS_GE_512-NEXT: ret
  %a = load <8 x double>, <8 x double>* %ap
  %b = load <8 x double>, <8 x double>* %bp
  %mask = fcmp oeq <8 x double> %a, %b
  %load = call <8 x double> @llvm.masked.load.v8f64(<8 x double>* %ap, i32 8, <8 x i1> %mask, <8 x double> %b)
  ret <8 x double> %load
}

define <32 x i16> @masked_load_sext_v32i8i16(<32 x i8>* %ap, <32 x i8>* %bp) #0 {
; CHECK-LABEL: masked_load_sext_v32i8i16:
; VBITS_GE_512: ptrue [[PG0:p[0-9]+]].b, vl32
; VBITS_GE_512-NEXT: ld1b { [[Z0:z[0-9]+]].b }, p0/z, [x0]
; VBITS_GE_512-NEXT: ld1b { [[Z1:z[0-9]+]].b }, p0/z, [x1]
; VBITS_GE_512-NEXT: cmpeq [[PG1:p[0-9]+]].b, [[PG0]]/z, [[Z0]].b, [[Z1]].b
; VBITS_GE_512-NEXT: ld1b { [[Z0]].b }, [[PG1]]/z, [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: sunpklo [[Z0]].h, [[Z0]].b
; VBITS_GE_512-NEXT: st1h { [[Z0]].h }, [[PG1]], [x8]
; VBITS_GE_512-NEXT: ret
  %a = load <32 x i8>, <32 x i8>* %ap
  %b = load <32 x i8>, <32 x i8>* %bp
  %mask = icmp eq <32 x i8> %a, %b
  %load = call <32 x i8> @llvm.masked.load.v32i8(<32 x i8>* %ap, i32 8, <32 x i1> %mask, <32 x i8> undef)
  %ext = sext <32 x i8> %load to <32 x i16>
  ret <32 x i16> %ext
}

define <16 x i32> @masked_load_sext_v16i8i32(<16 x i8>* %ap, <16 x i8>* %bp) #0 {
; CHECK-LABEL: masked_load_sext_v16i8i32:
; VBITS_GE_512: ldr q0, [x0]
; VBITS_GE_512-NEXT: ldr q1, [x1]
; VBITS_GE_512-NEXT: ptrue [[PG0:p[0-9]+]].b, vl16
; VBITS_GE_512-NEXT: cmeq v[[V:[0-9]+]].16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
; VBITS_GE_512-NEXT: cmpne [[PG2:p[0-9]+]].b, [[PG0]]/z, [[Z0]].b, #0
; VBITS_GE_512-NEXT: ld1b { [[Z0]].b }, [[PG2]]/z, [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: sunpklo [[Z0]].h, [[Z0]].b
; VBITS_GE_512-NEXT: sunpklo [[Z0]].s, [[Z0]].h
; VBITS_GE_512-NEXT: st1w { [[Z0]].s }, [[PG2]], [x8]
; VBITS_GE_512: ret
  %a = load <16 x i8>, <16 x i8>* %ap
  %b = load <16 x i8>, <16 x i8>* %bp
  %mask = icmp eq <16 x i8> %a, %b
  %load = call <16 x i8> @llvm.masked.load.v16i8(<16 x i8>* %ap, i32 8, <16 x i1> %mask, <16 x i8> undef)
  %ext = sext <16 x i8> %load to <16 x i32>
  ret <16 x i32> %ext
}

define <8 x i64> @masked_load_sext_v8i8i64(<8 x i8>* %ap, <8 x i8>* %bp) #0 {
; CHECK-LABEL: masked_load_sext_v8i8i64:
; VBITS_GE_512: ldr d0, [x0]
; VBITS_GE_512-NEXT: ldr d1, [x1]
; VBITS_GE_512-NEXT: ptrue [[PG0:p[0-9]+]].b, vl8
; VBITS_GE_512-NEXT: cmeq v[[V:[0-9]+]].8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
; VBITS_GE_512-NEXT: cmpne p[[PG:[0-9]+]].b, p0/z, z[[V]].b, #0
; VBITS_GE_512-NEXT: ld1b { [[Z0]].b }, p[[PG]]/z, [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: sunpklo [[Z0]].h, [[Z0]].b
; VBITS_GE_512-NEXT: sunpklo [[Z0]].s, [[Z0]].h
; VBITS_GE_512-NEXT: sunpklo [[Z0]].d, [[Z0]].s
; VBITS_GE_512-NEXT: st1d { [[Z0]].d }, [[PG2]], [x8]
; VBITS_GE_512-NEXT: ret
  %a = load <8 x i8>, <8 x i8>* %ap
  %b = load <8 x i8>, <8 x i8>* %bp
  %mask = icmp eq <8 x i8> %a, %b
  %load = call <8 x i8> @llvm.masked.load.v8i8(<8 x i8>* %ap, i32 8, <8 x i1> %mask, <8 x i8> undef)
  %ext = sext <8 x i8> %load to <8 x i64>
  ret <8 x i64> %ext
}

define <16 x i32> @masked_load_sext_v16i16i32(<16 x i16>* %ap, <16 x i16>* %bp) #0 {
; CHECK-LABEL: masked_load_sext_v16i16i32:
; VBITS_GE_512: ptrue [[PG0:p[0-9]+]].h, vl16
; VBITS_GE_512-NEXT: ld1h { [[Z0:z[0-9]+]].h }, p0/z, [x0]
; VBITS_GE_512-NEXT: ld1h { [[Z1:z[0-9]+]].h }, p0/z, [x1]
; VBITS_GE_512-NEXT: cmpeq [[PG1:p[0-9]+]].h, [[PG0]]/z, [[Z0]].h, [[Z1]].h
; VBITS_GE_512-NEXT: ld1h { [[Z0]].h }, [[PG1]]/z, [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: sunpklo [[Z0]].s, [[Z0]].h
; VBITS_GE_512-NEXT: st1w { [[Z0]].s }, [[PG1]], [x8]
; VBITS_GE_512-NEXT: ret
  %a = load <16 x i16>, <16 x i16>* %ap
  %b = load <16 x i16>, <16 x i16>* %bp
  %mask = icmp eq <16 x i16> %a, %b
  %load = call <16 x i16> @llvm.masked.load.v16i16(<16 x i16>* %ap, i32 8, <16 x i1> %mask, <16 x i16> undef)
  %ext = sext <16 x i16> %load to <16 x i32>
  ret <16 x i32> %ext
}

define <8 x i64> @masked_load_sext_v8i16i64(<8 x i16>* %ap, <8 x i16>* %bp) #0 {
; CHECK-LABEL: masked_load_sext_v8i16i64:
; VBITS_GE_512: ldr q0, [x0]
; VBITS_GE_512-NEXT: ldr q1, [x1]
; VBITS_GE_512-NEXT: ptrue [[PG0:p[0-9]+]].h, vl8
; VBITS_GE_512-NEXT: cmeq v[[V:[0-9]+]].8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
; VBITS_GE_512-NEXT: cmpne p[[PG:[0-9]+]].h, p0/z, z[[V]].h, #0
; VBITS_GE_512-NEXT: ld1h { [[Z0]].h }, p[[PG]]/z, [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: sunpklo [[Z0]].s, [[Z0]].h
; VBITS_GE_512-NEXT: sunpklo [[Z0]].d, [[Z0]].s
; VBITS_GE_512-NEXT: st1d { [[Z0]].d }, [[PG2]], [x8]
; VBITS_GE_512-NEXT: ret
  %a = load <8 x i16>, <8 x i16>* %ap
  %b = load <8 x i16>, <8 x i16>* %bp
  %mask = icmp eq <8 x i16> %a, %b
  %load = call <8 x i16> @llvm.masked.load.v8i16(<8 x i16>* %ap, i32 8, <8 x i1> %mask, <8 x i16> undef)
  %ext = sext <8 x i16> %load to <8 x i64>
  ret <8 x i64> %ext
}

define <8 x i64> @masked_load_sext_v8i32i64(<8 x i32>* %ap, <8 x i32>* %bp) #0 {
; CHECK-LABEL: masked_load_sext_v8i32i64:
; VBITS_GE_512: ptrue [[PG0:p[0-9]+]].s, vl8
; VBITS_GE_512-NEXT: ld1w { [[Z0:z[0-9]+]].s }, p0/z, [x0]
; VBITS_GE_512-NEXT: ld1w { [[Z1:z[0-9]+]].s }, p0/z, [x1]
; VBITS_GE_512-NEXT: cmpeq [[PG1:p[0-9]+]].s, [[PG0]]/z, [[Z0]].s, [[Z1]].s
; VBITS_GE_512-NEXT: ld1w { [[Z0]].s }, [[PG1]]/z, [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: sunpklo [[Z0]].d, [[Z0]].s
; VBITS_GE_512-NEXT: st1d { [[Z0]].d }, [[PG1]], [x8]
; VBITS_GE_512-NEXT: ret
  %a = load <8 x i32>, <8 x i32>* %ap
  %b = load <8 x i32>, <8 x i32>* %bp
  %mask = icmp eq <8 x i32> %a, %b
  %load = call <8 x i32> @llvm.masked.load.v8i32(<8 x i32>* %ap, i32 8, <8 x i1> %mask, <8 x i32> undef)
  %ext = sext <8 x i32> %load to <8 x i64>
  ret <8 x i64> %ext
}

define <32 x i16> @masked_load_zext_v32i8i16(<32 x i8>* %ap, <32 x i8>* %bp) #0 {
; CHECK-LABEL: masked_load_zext_v32i8i16:
; VBITS_GE_512: ptrue [[PG0:p[0-9]+]].b, vl32
; VBITS_GE_512-NEXT: ld1b { [[Z0:z[0-9]+]].b }, p0/z, [x0]
; VBITS_GE_512-NEXT: ld1b { [[Z1:z[0-9]+]].b }, p0/z, [x1]
; VBITS_GE_512-NEXT: cmpeq [[PG1:p[0-9]+]].b, [[PG0]]/z, [[Z0]].b, [[Z1]].b
; VBITS_GE_512-NEXT: ld1b { [[Z0]].b }, [[PG1]]/z, [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: uunpklo [[Z0]].h, [[Z0]].b
; VBITS_GE_512-NEXT: st1h { [[Z0]].h }, [[PG1]], [x8]
; VBITS_GE_512-NEXT: ret
  %a = load <32 x i8>, <32 x i8>* %ap
  %b = load <32 x i8>, <32 x i8>* %bp
  %mask = icmp eq <32 x i8> %a, %b
  %load = call <32 x i8> @llvm.masked.load.v32i8(<32 x i8>* %ap, i32 8, <32 x i1> %mask, <32 x i8> undef)
  %ext = zext <32 x i8> %load to <32 x i16>
  ret <32 x i16> %ext
}

define <16 x i32> @masked_load_zext_v16i8i32(<16 x i8>* %ap, <16 x i8>* %bp) #0 {
; CHECK-LABEL: masked_load_zext_v16i8i32:
; VBITS_GE_512: ldr q0, [x0]
; VBITS_GE_512-NEXT: ldr q1, [x1]
; VBITS_GE_512-NEXT: ptrue [[PG0:p[0-9]+]].b, vl16
; VBITS_GE_512-NEXT: cmeq v[[V:[0-9]+]].16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
; VBITS_GE_512-NEXT: cmpne [[PG2:p[0-9]+]].b, [[PG0]]/z, [[Z0]].b, #0
; VBITS_GE_512-NEXT: ld1b { [[Z0]].b }, [[PG2]]/z, [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: uunpklo [[Z0]].h, [[Z0]].b
; VBITS_GE_512-NEXT: uunpklo [[Z0]].s, [[Z0]].h
; VBITS_GE_512-NEXT: st1w { [[Z0]].s }, [[PG2]], [x8]
; VBITS_GE_512: ret
  %a = load <16 x i8>, <16 x i8>* %ap
  %b = load <16 x i8>, <16 x i8>* %bp
  %mask = icmp eq <16 x i8> %a, %b
  %load = call <16 x i8> @llvm.masked.load.v16i8(<16 x i8>* %ap, i32 8, <16 x i1> %mask, <16 x i8> undef)
  %ext = zext <16 x i8> %load to <16 x i32>
  ret <16 x i32> %ext
}

define <8 x i64> @masked_load_zext_v8i8i64(<8 x i8>* %ap, <8 x i8>* %bp) #0 {
; CHECK-LABEL: masked_load_zext_v8i8i64:
; VBITS_GE_512: ldr d0, [x0]
; VBITS_GE_512-NEXT: ldr d1, [x1]
; VBITS_GE_512-NEXT: ptrue [[PG0:p[0-9]+]].b, vl8
; VBITS_GE_512-NEXT: cmeq v[[V:[0-9]+]].8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
; VBITS_GE_512-NEXT: cmpne p[[PG:[0-9]+]].b, p0/z, z[[V]].b, #0
; VBITS_GE_512-NEXT: ld1b { [[Z0]].b }, p[[PG]]/z, [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: uunpklo [[Z0]].h, [[Z0]].b
; VBITS_GE_512-NEXT: uunpklo [[Z0]].s, [[Z0]].h
; VBITS_GE_512-NEXT: uunpklo [[Z0]].d, [[Z0]].s
; VBITS_GE_512-NEXT: st1d { [[Z0]].d }, [[PG2]], [x8]
; VBITS_GE_512-NEXT: ret
  %a = load <8 x i8>, <8 x i8>* %ap
  %b = load <8 x i8>, <8 x i8>* %bp
  %mask = icmp eq <8 x i8> %a, %b
  %load = call <8 x i8> @llvm.masked.load.v8i8(<8 x i8>* %ap, i32 8, <8 x i1> %mask, <8 x i8> undef)
  %ext = zext <8 x i8> %load to <8 x i64>
  ret <8 x i64> %ext
}

define <16 x i32> @masked_load_zext_v16i16i32(<16 x i16>* %ap, <16 x i16>* %bp) #0 {
; CHECK-LABEL: masked_load_zext_v16i16i32:
; VBITS_GE_512: ptrue [[PG0:p[0-9]+]].h, vl16
; VBITS_GE_512-NEXT: ld1h { [[Z0:z[0-9]+]].h }, p0/z, [x0]
; VBITS_GE_512-NEXT: ld1h { [[Z1:z[0-9]+]].h }, p0/z, [x1]
; VBITS_GE_512-NEXT: cmpeq [[PG1:p[0-9]+]].h, [[PG0]]/z, [[Z0]].h, [[Z1]].h
; VBITS_GE_512-NEXT: ld1h { [[Z0]].h }, [[PG1]]/z, [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: uunpklo [[Z0]].s, [[Z0]].h
; VBITS_GE_512-NEXT: st1w { [[Z0]].s }, [[PG1]], [x8]
; VBITS_GE_512-NEXT: ret
  %a = load <16 x i16>, <16 x i16>* %ap
  %b = load <16 x i16>, <16 x i16>* %bp
  %mask = icmp eq <16 x i16> %a, %b
  %load = call <16 x i16> @llvm.masked.load.v16i16(<16 x i16>* %ap, i32 8, <16 x i1> %mask, <16 x i16> undef)
  %ext = zext <16 x i16> %load to <16 x i32>
  ret <16 x i32> %ext
}

define <8 x i64> @masked_load_zext_v8i16i64(<8 x i16>* %ap, <8 x i16>* %bp) #0 {
; CHECK-LABEL: masked_load_zext_v8i16i64:
; VBITS_GE_512: ldr q0, [x0]
; VBITS_GE_512-NEXT: ldr q1, [x1]
; VBITS_GE_512-NEXT: ptrue [[PG0:p[0-9]+]].h, vl8
; VBITS_GE_512-NEXT: cmeq v[[V:[0-9]+]].8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
; VBITS_GE_512-NEXT: cmpne p[[PG:[0-9]+]].h, p0/z, z[[V]].h, #0
; VBITS_GE_512-NEXT: ld1h { [[Z0]].h }, p[[PG]]/z, [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: uunpklo [[Z0]].s, [[Z0]].h
; VBITS_GE_512-NEXT: uunpklo [[Z0]].d, [[Z0]].s
; VBITS_GE_512-NEXT: st1d { [[Z0]].d }, [[PG2]], [x8]
; VBITS_GE_512-NEXT: ret
  %a = load <8 x i16>, <8 x i16>* %ap
  %b = load <8 x i16>, <8 x i16>* %bp
  %mask = icmp eq <8 x i16> %a, %b
  %load = call <8 x i16> @llvm.masked.load.v8i16(<8 x i16>* %ap, i32 8, <8 x i1> %mask, <8 x i16> undef)
  %ext = zext <8 x i16> %load to <8 x i64>
  ret <8 x i64> %ext
}

define <8 x i64> @masked_load_zext_v8i32i64(<8 x i32>* %ap, <8 x i32>* %bp) #0 {
; CHECK-LABEL: masked_load_zext_v8i32i64:
; VBITS_GE_512: ptrue [[PG0:p[0-9]+]].s, vl8
; VBITS_GE_512-NEXT: ld1w { [[Z0:z[0-9]+]].s }, p0/z, [x0]
; VBITS_GE_512-NEXT: ld1w { [[Z1:z[0-9]+]].s }, p0/z, [x1]
; VBITS_GE_512-NEXT: cmpeq [[PG1:p[0-9]+]].s, [[PG0]]/z, [[Z0]].s, [[Z1]].s
; VBITS_GE_512-NEXT: ld1w { [[Z0]].s }, [[PG1]]/z, [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: uunpklo [[Z0]].d, [[Z0]].s
; VBITS_GE_512-NEXT: st1d { [[Z0]].d }, [[PG1]], [x8]
; VBITS_GE_512-NEXT: ret
  %a = load <8 x i32>, <8 x i32>* %ap
  %b = load <8 x i32>, <8 x i32>* %bp
  %mask = icmp eq <8 x i32> %a, %b
  %load = call <8 x i32> @llvm.masked.load.v8i32(<8 x i32>* %ap, i32 8, <8 x i1> %mask, <8 x i32> undef)
  %ext = zext <8 x i32> %load to <8 x i64>
  ret <8 x i64> %ext
}

declare <2 x half> @llvm.masked.load.v2f16(<2 x half>*, i32, <2 x i1>, <2 x half>)
declare <2 x float> @llvm.masked.load.v2f32(<2 x float>*, i32, <2 x i1>, <2 x float>)
declare <4 x float> @llvm.masked.load.v4f32(<4 x float>*, i32, <4 x i1>, <4 x float>)
declare <8 x float> @llvm.masked.load.v8f32(<8 x float>*, i32, <8 x i1>, <8 x float>)
declare <16 x float> @llvm.masked.load.v16f32(<16 x float>*, i32, <16 x i1>, <16 x float>)
declare <32 x float> @llvm.masked.load.v32f32(<32 x float>*, i32, <32 x i1>, <32 x float>)
declare <64 x float> @llvm.masked.load.v64f32(<64 x float>*, i32, <64 x i1>, <64 x float>)

declare <64 x i8> @llvm.masked.load.v64i8(<64 x i8>*, i32, <64 x i1>, <64 x i8>)
declare <32 x i8> @llvm.masked.load.v32i8(<32 x i8>*, i32, <32 x i1>, <32 x i8>)
declare <16 x i8> @llvm.masked.load.v16i8(<16 x i8>*, i32, <16 x i1>, <16 x i8>)
declare <16 x i16> @llvm.masked.load.v16i16(<16 x i16>*, i32, <16 x i1>, <16 x i16>)
declare <8 x i8> @llvm.masked.load.v8i8(<8 x i8>*, i32, <8 x i1>, <8 x i8>)
declare <8 x i16> @llvm.masked.load.v8i16(<8 x i16>*, i32, <8 x i1>, <8 x i16>)
declare <8 x i32> @llvm.masked.load.v8i32(<8 x i32>*, i32, <8 x i1>, <8 x i32>)
declare <32 x i16> @llvm.masked.load.v32i16(<32 x i16>*, i32, <32 x i1>, <32 x i16>)
declare <16 x i32> @llvm.masked.load.v16i32(<16 x i32>*, i32, <16 x i1>, <16 x i32>)
declare <8 x i64> @llvm.masked.load.v8i64(<8 x i64>*, i32, <8 x i1>, <8 x i64>)
declare <8 x double> @llvm.masked.load.v8f64(<8 x double>*, i32, <8 x i1>, <8 x double>)

attributes #0 = { "target-features"="+sve" }
