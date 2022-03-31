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
; UADDV
;

; Don't use SVE for 64-bit vectors.
define i8 @uaddv_v8i8(<8 x i8> %a) #0 {
; CHECK-LABEL: uaddv_v8i8:
; CHECK: addv b0, v0.8b
; CHECK: ret
  %res = call i8 @llvm.vector.reduce.add.v8i8(<8 x i8> %a)
  ret i8 %res
}

; Don't use SVE for 128-bit vectors.
define i8 @uaddv_v16i8(<16 x i8> %a) #0 {
; CHECK-LABEL: uaddv_v16i8:
; CHECK: addv b0, v0.16b
; CHECK: ret
  %res = call i8 @llvm.vector.reduce.add.v16i8(<16 x i8> %a)
  ret i8 %res
}

define i8 @uaddv_v32i8(<32 x i8>* %a) #0 {
; CHECK-LABEL: uaddv_v32i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl32
; CHECK-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].b
; CHECK-NEXT: fmov x0, [[REDUCE]]
; CHECK-NEXT: ret
  %op = load <32 x i8>, <32 x i8>* %a
  %res = call i8 @llvm.vector.reduce.add.v32i8(<32 x i8> %op)
  ret i8 %res
}

define i8 @uaddv_v64i8(<64 x i8>* %a) #0 {
; CHECK-LABEL: uaddv_v64i8:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].b, vl64
; VBITS_GE_512-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_512-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].b, vl32
; VBITS_EQ_256-DAG: mov w[[NUMELTS:[0-9]+]], #32
; VBITS_EQ_256-DAG: ld1b { [[LO:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1b { [[HI:z[0-9]+]].b }, [[PG]]/z, [x0, x[[NUMELTS]]]
; VBITS_EQ_256-DAG: add [[ADD:z[0-9]+]].b, [[LO]].b, [[HI]].b
; VBITS_EQ_256-DAG: addv [[REDUCE:d[0-9]+]], [[PG]], [[ADD]].b
; VBITS_EQ_256-NEXT: fmov x0, [[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <64 x i8>, <64 x i8>* %a
  %res = call i8 @llvm.vector.reduce.add.v64i8(<64 x i8> %op)
  ret i8 %res
}

define i8 @uaddv_v128i8(<128 x i8>* %a) #0 {
; CHECK-LABEL: uaddv_v128i8:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].b, vl128
; VBITS_GE_1024-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_1024-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <128 x i8>, <128 x i8>* %a
  %res = call i8 @llvm.vector.reduce.add.v128i8(<128 x i8> %op)
  ret i8 %res
}

define i8 @uaddv_v256i8(<256 x i8>* %a) #0 {
; CHECK-LABEL: uaddv_v256i8:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].b, vl256
; VBITS_GE_2048-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_2048-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <256 x i8>, <256 x i8>* %a
  %res = call i8 @llvm.vector.reduce.add.v256i8(<256 x i8> %op)
  ret i8 %res
}

; Don't use SVE for 64-bit vectors.
define i16 @uaddv_v4i16(<4 x i16> %a) #0 {
; CHECK-LABEL: uaddv_v4i16:
; CHECK: addv h0, v0.4h
; CHECK: ret
  %res = call i16 @llvm.vector.reduce.add.v4i16(<4 x i16> %a)
  ret i16 %res
}

; Don't use SVE for 128-bit vectors.
define i16 @uaddv_v8i16(<8 x i16> %a) #0 {
; CHECK-LABEL: uaddv_v8i16:
; CHECK: addv h0, v0.8h
; CHECK: ret
  %res = call i16 @llvm.vector.reduce.add.v8i16(<8 x i16> %a)
  ret i16 %res
}

define i16 @uaddv_v16i16(<16 x i16>* %a) #0 {
; CHECK-LABEL: uaddv_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].h
; CHECK-NEXT: fmov x0, [[REDUCE]]
; CHECK-NEXT: ret
  %op = load <16 x i16>, <16 x i16>* %a
  %res = call i16 @llvm.vector.reduce.add.v16i16(<16 x i16> %op)
  ret i16 %res
}

define i16 @uaddv_v32i16(<32 x i16>* %a) #0 {
; CHECK-LABEL: uaddv_v32i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_512-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #16
; VBITS_EQ_256-DAG: ld1h { [[LO:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[HI:z[0-9]+]].h }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #1]
; VBITS_EQ_256-DAG: add [[ADD:z[0-9]+]].h, [[LO]].h, [[HI]].h
; VBITS_EQ_256-DAG: addv [[REDUCE:d[0-9]+]], [[PG]], [[ADD]].h
; VBITS_EQ_256-NEXT: fmov x0, [[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <32 x i16>, <32 x i16>* %a
  %res = call i16 @llvm.vector.reduce.add.v32i16(<32 x i16> %op)
  ret i16 %res
}

define i16 @uaddv_v64i16(<64 x i16>* %a) #0 {
; CHECK-LABEL: uaddv_v64i16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_1024-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <64 x i16>, <64 x i16>* %a
  %res = call i16 @llvm.vector.reduce.add.v64i16(<64 x i16> %op)
  ret i16 %res
}

define i16 @uaddv_v128i16(<128 x i16>* %a) #0 {
; CHECK-LABEL: uaddv_v128i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_2048-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <128 x i16>, <128 x i16>* %a
  %res = call i16 @llvm.vector.reduce.add.v128i16(<128 x i16> %op)
  ret i16 %res
}

; Don't use SVE for 64-bit vectors.
define i32 @uaddv_v2i32(<2 x i32> %a) #0 {
; CHECK-LABEL: uaddv_v2i32:
; CHECK: addp v0.2s, v0.2s
; CHECK: ret
  %res = call i32 @llvm.vector.reduce.add.v2i32(<2 x i32> %a)
  ret i32 %res
}

; Don't use SVE for 128-bit vectors.
define i32 @uaddv_v4i32(<4 x i32> %a) #0 {
; CHECK-LABEL: uaddv_v4i32:
; CHECK: addv s0, v0.4s
; CHECK: ret
  %res = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %a)
  ret i32 %res
}

define i32 @uaddv_v8i32(<8 x i32>* %a) #0 {
; CHECK-LABEL: uaddv_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].s
; CHECK-NEXT: fmov x0, [[REDUCE]]
; CHECK-NEXT: ret
  %op = load <8 x i32>, <8 x i32>* %a
  %res = call i32 @llvm.vector.reduce.add.v8i32(<8 x i32> %op)
  ret i32 %res
}

define i32 @uaddv_v16i32(<16 x i32>* %a) #0 {
; CHECK-LABEL: uaddv_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_512-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #8
; VBITS_EQ_256-DAG: ld1w { [[LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[HI:z[0-9]+]].s }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #2]
; VBITS_EQ_256-DAG: add [[ADD:z[0-9]+]].s, [[LO]].s, [[HI]].s
; VBITS_EQ_256-DAG: addv [[REDUCE:d[0-9]+]], [[PG]], [[ADD]].s
; VBITS_EQ_256-NEXT: fmov x0, [[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <16 x i32>, <16 x i32>* %a
  %res = call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %op)
  ret i32 %res
}

define i32 @uaddv_v32i32(<32 x i32>* %a) #0 {
; CHECK-LABEL: uaddv_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_1024-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <32 x i32>, <32 x i32>* %a
  %res = call i32 @llvm.vector.reduce.add.v32i32(<32 x i32> %op)
  ret i32 %res
}

define i32 @uaddv_v64i32(<64 x i32>* %a) #0 {
; CHECK-LABEL: uaddv_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_2048-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <64 x i32>, <64 x i32>* %a
  %res = call i32 @llvm.vector.reduce.add.v64i32(<64 x i32> %op)
  ret i32 %res
}

; Nothing to do for single element vectors.
define i64 @uaddv_v1i64(<1 x i64> %a) #0 {
; CHECK-LABEL: uaddv_v1i64:
; CHECK: fmov x0, d0
; CHECK: ret
  %res = call i64 @llvm.vector.reduce.add.v1i64(<1 x i64> %a)
  ret i64 %res
}

; Don't use SVE for 128-bit vectors.
define i64 @uaddv_v2i64(<2 x i64> %a) #0 {
; CHECK-LABEL: uaddv_v2i64:
; CHECK: addp d0, v0.2d
; CHECK: ret
  %res = call i64 @llvm.vector.reduce.add.v2i64(<2 x i64> %a)
  ret i64 %res
}

define i64 @uaddv_v4i64(<4 x i64>* %a) #0 {
; CHECK-LABEL: uaddv_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; CHECK-NEXT: fmov x0, [[REDUCE]]
; CHECK-NEXT: ret
  %op = load <4 x i64>, <4 x i64>* %a
  %res = call i64 @llvm.vector.reduce.add.v4i64(<4 x i64> %op)
  ret i64 %res
}

define i64 @uaddv_v8i64(<8 x i64>* %a) #0 {
; CHECK-LABEL: uaddv_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_512-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: ld1d { [[LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[HI:z[0-9]+]].d }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-DAG: add [[ADD:z[0-9]+]].d, [[LO]].d, [[HI]].d
; VBITS_EQ_256-DAG: addv [[REDUCE:d[0-9]+]], [[PG]], [[ADD]].d
; VBITS_EQ_256-NEXT: fmov x0, [[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <8 x i64>, <8 x i64>* %a
  %res = call i64 @llvm.vector.reduce.add.v8i64(<8 x i64> %op)
  ret i64 %res
}

define i64 @uaddv_v16i64(<16 x i64>* %a) #0 {
; CHECK-LABEL: uaddv_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_1024-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <16 x i64>, <16 x i64>* %a
  %res = call i64 @llvm.vector.reduce.add.v16i64(<16 x i64> %op)
  ret i64 %res
}

define i64 @uaddv_v32i64(<32 x i64>* %a) #0 {
; CHECK-LABEL: uaddv_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_2048-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <32 x i64>, <32 x i64>* %a
  %res = call i64 @llvm.vector.reduce.add.v32i64(<32 x i64> %op)
  ret i64 %res
}

;
; SMAXV
;

; Don't use SVE for 64-bit vectors.
define i8 @smaxv_v8i8(<8 x i8> %a) #0 {
; CHECK-LABEL: smaxv_v8i8:
; CHECK: smaxv b0, v0.8b
; CHECK: ret
  %res = call i8 @llvm.vector.reduce.smax.v8i8(<8 x i8> %a)
  ret i8 %res
}

; Don't use SVE for 128-bit vectors.
define i8 @smaxv_v16i8(<16 x i8> %a) #0 {
; CHECK-LABEL: smaxv_v16i8:
; CHECK: smaxv b0, v0.16b
; CHECK: ret
  %res = call i8 @llvm.vector.reduce.smax.v16i8(<16 x i8> %a)
  ret i8 %res
}

define i8 @smaxv_v32i8(<32 x i8>* %a) #0 {
; CHECK-LABEL: smaxv_v32i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl32
; CHECK-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-NEXT: smaxv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; CHECK-NEXT: fmov w0, s[[REDUCE]]
; CHECK-NEXT: ret
  %op = load <32 x i8>, <32 x i8>* %a
  %res = call i8 @llvm.vector.reduce.smax.v32i8(<32 x i8> %op)
  ret i8 %res
}

define i8 @smaxv_v64i8(<64 x i8>* %a) #0 {
; CHECK-LABEL: smaxv_v64i8:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].b, vl64
; VBITS_GE_512-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: smaxv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_512-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].b, vl32
; VBITS_EQ_256-DAG: mov w[[NUMELTS:[0-9]+]], #32
; VBITS_EQ_256-DAG: ld1b { [[LO:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1b { [[HI:z[0-9]+]].b }, [[PG]]/z, [x0, x[[NUMELTS]]]
; VBITS_EQ_256-DAG: smax [[MAX:z[0-9]+]].b, [[PG]]/m, [[HI]].b, [[LO]].b
; VBITS_EQ_256-DAG: smaxv b[[REDUCE:[0-9]+]], [[PG]], [[MAX]].b
; VBITS_EQ_256-NEXT: fmov w0, s[[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <64 x i8>, <64 x i8>* %a
  %res = call i8 @llvm.vector.reduce.smax.v64i8(<64 x i8> %op)
  ret i8 %res
}

define i8 @smaxv_v128i8(<128 x i8>* %a) #0 {
; CHECK-LABEL: smaxv_v128i8:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].b, vl128
; VBITS_GE_1024-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: smaxv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_1024-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <128 x i8>, <128 x i8>* %a
  %res = call i8 @llvm.vector.reduce.smax.v128i8(<128 x i8> %op)
  ret i8 %res
}

define i8 @smaxv_v256i8(<256 x i8>* %a) #0 {
; CHECK-LABEL: smaxv_v256i8:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].b, vl256
; VBITS_GE_2048-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: smaxv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_2048-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <256 x i8>, <256 x i8>* %a
  %res = call i8 @llvm.vector.reduce.smax.v256i8(<256 x i8> %op)
  ret i8 %res
}

; Don't use SVE for 64-bit vectors.
define i16 @smaxv_v4i16(<4 x i16> %a) #0 {
; CHECK-LABEL: smaxv_v4i16:
; CHECK: smaxv h0, v0.4h
; CHECK: ret
  %res = call i16 @llvm.vector.reduce.smax.v4i16(<4 x i16> %a)
  ret i16 %res
}

; Don't use SVE for 128-bit vectors.
define i16 @smaxv_v8i16(<8 x i16> %a) #0 {
; CHECK-LABEL: smaxv_v8i16:
; CHECK: smaxv h0, v0.8h
; CHECK: ret
  %res = call i16 @llvm.vector.reduce.smax.v8i16(<8 x i16> %a)
  ret i16 %res
}

define i16 @smaxv_v16i16(<16 x i16>* %a) #0 {
; CHECK-LABEL: smaxv_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-NEXT: smaxv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; CHECK-NEXT: fmov w0, s[[REDUCE]]
; CHECK-NEXT: ret
  %op = load <16 x i16>, <16 x i16>* %a
  %res = call i16 @llvm.vector.reduce.smax.v16i16(<16 x i16> %op)
  ret i16 %res
}

define i16 @smaxv_v32i16(<32 x i16>* %a) #0 {
; CHECK-LABEL: smaxv_v32i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: smaxv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_512-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #16
; VBITS_EQ_256-DAG: ld1h { [[LO:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[HI:z[0-9]+]].h }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #1]
; VBITS_EQ_256-DAG: smax [[MAX:z[0-9]+]].h, [[PG]]/m, [[HI]].h, [[LO]].h
; VBITS_EQ_256-DAG: smaxv h[[REDUCE:[0-9]+]], [[PG]], [[MAX]].h
; VBITS_EQ_256-NEXT: fmov w0, s[[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <32 x i16>, <32 x i16>* %a
  %res = call i16 @llvm.vector.reduce.smax.v32i16(<32 x i16> %op)
  ret i16 %res
}

define i16 @smaxv_v64i16(<64 x i16>* %a) #0 {
; CHECK-LABEL: smaxv_v64i16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: smaxv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_1024-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <64 x i16>, <64 x i16>* %a
  %res = call i16 @llvm.vector.reduce.smax.v64i16(<64 x i16> %op)
  ret i16 %res
}

define i16 @smaxv_v128i16(<128 x i16>* %a) #0 {
; CHECK-LABEL: smaxv_v128i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: smaxv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_2048-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <128 x i16>, <128 x i16>* %a
  %res = call i16 @llvm.vector.reduce.smax.v128i16(<128 x i16> %op)
  ret i16 %res
}

; Don't use SVE for 64-bit vectors.
define i32 @smaxv_v2i32(<2 x i32> %a) #0 {
; CHECK-LABEL: smaxv_v2i32:
; CHECK: smaxp v0.2s, v0.2s
; CHECK: ret
  %res = call i32 @llvm.vector.reduce.smax.v2i32(<2 x i32> %a)
  ret i32 %res
}

; Don't use SVE for 128-bit vectors.
define i32 @smaxv_v4i32(<4 x i32> %a) #0 {
; CHECK-LABEL: smaxv_v4i32:
; CHECK: smaxv s0, v0.4s
; CHECK: ret
  %res = call i32 @llvm.vector.reduce.smax.v4i32(<4 x i32> %a)
  ret i32 %res
}

define i32 @smaxv_v8i32(<8 x i32>* %a) #0 {
; CHECK-LABEL: smaxv_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: smaxv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; CHECK-NEXT: fmov w0, [[REDUCE]]
; CHECK-NEXT: ret
  %op = load <8 x i32>, <8 x i32>* %a
  %res = call i32 @llvm.vector.reduce.smax.v8i32(<8 x i32> %op)
  ret i32 %res
}

define i32 @smaxv_v16i32(<16 x i32>* %a) #0 {
; CHECK-LABEL: smaxv_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: smaxv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_512-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #8
; VBITS_EQ_256-DAG: ld1w { [[LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[HI:z[0-9]+]].s }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #2]
; VBITS_EQ_256-DAG: smax [[MAX:z[0-9]+]].s, [[PG]]/m, [[HI]].s, [[LO]].s
; VBITS_EQ_256-DAG: smaxv [[REDUCE:s[0-9]+]], [[PG]], [[MAX]].s
; VBITS_EQ_256-NEXT: fmov w0, [[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <16 x i32>, <16 x i32>* %a
  %res = call i32 @llvm.vector.reduce.smax.v16i32(<16 x i32> %op)
  ret i32 %res
}

define i32 @smaxv_v32i32(<32 x i32>* %a) #0 {
; CHECK-LABEL: smaxv_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: smaxv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_1024-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <32 x i32>, <32 x i32>* %a
  %res = call i32 @llvm.vector.reduce.smax.v32i32(<32 x i32> %op)
  ret i32 %res
}

define i32 @smaxv_v64i32(<64 x i32>* %a) #0 {
; CHECK-LABEL: smaxv_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: smaxv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_2048-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <64 x i32>, <64 x i32>* %a
  %res = call i32 @llvm.vector.reduce.smax.v64i32(<64 x i32> %op)
  ret i32 %res
}

; Nothing to do for single element vectors.
define i64 @smaxv_v1i64(<1 x i64> %a) #0 {
; CHECK-LABEL: smaxv_v1i64:
; CHECK: fmov x0, d0
; CHECK: ret
  %res = call i64 @llvm.vector.reduce.smax.v1i64(<1 x i64> %a)
  ret i64 %res
}

; No NEON 64-bit vector SMAXV support. Use SVE.
define i64 @smaxv_v2i64(<2 x i64> %a) #0 {
; CHECK-LABEL: smaxv_v2i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl2
; CHECK-NEXT: smaxv [[REDUCE:d[0-9]+]], [[PG]], z0.d
; CHECK-NEXT: fmov x0, [[REDUCE]]
; CHECK-NEXT: ret
  %res = call i64 @llvm.vector.reduce.smax.v2i64(<2 x i64> %a)
  ret i64 %res
}

define i64 @smaxv_v4i64(<4 x i64>* %a) #0 {
; CHECK-LABEL: smaxv_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: smaxv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; CHECK-NEXT: fmov x0, [[REDUCE]]
; CHECK-NEXT: ret
  %op = load <4 x i64>, <4 x i64>* %a
  %res = call i64 @llvm.vector.reduce.smax.v4i64(<4 x i64> %op)
  ret i64 %res
}

define i64 @smaxv_v8i64(<8 x i64>* %a) #0 {
; CHECK-LABEL: smaxv_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: smaxv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_512-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: ld1d { [[LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[HI:z[0-9]+]].d }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-DAG: smax [[MAX:z[0-9]+]].d, [[PG]]/m, [[HI]].d, [[LO]].d
; VBITS_EQ_256-DAG: smaxv [[REDUCE:d[0-9]+]], [[PG]], [[MAX]].d
; VBITS_EQ_256-NEXT: fmov x0, [[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <8 x i64>, <8 x i64>* %a
  %res = call i64 @llvm.vector.reduce.smax.v8i64(<8 x i64> %op)
  ret i64 %res
}

define i64 @smaxv_v16i64(<16 x i64>* %a) #0 {
; CHECK-LABEL: smaxv_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: smaxv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_1024-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <16 x i64>, <16 x i64>* %a
  %res = call i64 @llvm.vector.reduce.smax.v16i64(<16 x i64> %op)
  ret i64 %res
}

define i64 @smaxv_v32i64(<32 x i64>* %a) #0 {
; CHECK-LABEL: smaxv_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: smaxv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_2048-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <32 x i64>, <32 x i64>* %a
  %res = call i64 @llvm.vector.reduce.smax.v32i64(<32 x i64> %op)
  ret i64 %res
}

;
; SMINV
;

; Don't use SVE for 64-bit vectors.
define i8 @sminv_v8i8(<8 x i8> %a) #0 {
; CHECK-LABEL: sminv_v8i8:
; CHECK: sminv b0, v0.8b
; CHECK: ret
  %res = call i8 @llvm.vector.reduce.smin.v8i8(<8 x i8> %a)
  ret i8 %res
}

; Don't use SVE for 128-bit vectors.
define i8 @sminv_v16i8(<16 x i8> %a) #0 {
; CHECK-LABEL: sminv_v16i8:
; CHECK: sminv b0, v0.16b
; CHECK: ret
  %res = call i8 @llvm.vector.reduce.smin.v16i8(<16 x i8> %a)
  ret i8 %res
}

define i8 @sminv_v32i8(<32 x i8>* %a) #0 {
; CHECK-LABEL: sminv_v32i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl32
; CHECK-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-NEXT: sminv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; CHECK-NEXT: fmov w0, s[[REDUCE]]
; CHECK-NEXT: ret
  %op = load <32 x i8>, <32 x i8>* %a
  %res = call i8 @llvm.vector.reduce.smin.v32i8(<32 x i8> %op)
  ret i8 %res
}

define i8 @sminv_v64i8(<64 x i8>* %a) #0 {
; CHECK-LABEL: sminv_v64i8:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].b, vl64
; VBITS_GE_512-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: sminv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_512-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].b, vl32
; VBITS_EQ_256-DAG: mov w[[NUMELTS:[0-9]+]], #32
; VBITS_EQ_256-DAG: ld1b { [[LO:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1b { [[HI:z[0-9]+]].b }, [[PG]]/z, [x0, x[[NUMELTS]]]
; VBITS_EQ_256-DAG: smin [[MIN:z[0-9]+]].b, [[PG]]/m, [[HI]].b, [[LO]].b
; VBITS_EQ_256-DAG: sminv b[[REDUCE:[0-9]+]], [[PG]], [[MIN]].b
; VBITS_EQ_256-NEXT: fmov w0, s[[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <64 x i8>, <64 x i8>* %a
  %res = call i8 @llvm.vector.reduce.smin.v64i8(<64 x i8> %op)
  ret i8 %res
}

define i8 @sminv_v128i8(<128 x i8>* %a) #0 {
; CHECK-LABEL: sminv_v128i8:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].b, vl128
; VBITS_GE_1024-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: sminv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_1024-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <128 x i8>, <128 x i8>* %a
  %res = call i8 @llvm.vector.reduce.smin.v128i8(<128 x i8> %op)
  ret i8 %res
}

define i8 @sminv_v256i8(<256 x i8>* %a) #0 {
; CHECK-LABEL: sminv_v256i8:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].b, vl256
; VBITS_GE_2048-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: sminv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_2048-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <256 x i8>, <256 x i8>* %a
  %res = call i8 @llvm.vector.reduce.smin.v256i8(<256 x i8> %op)
  ret i8 %res
}

; Don't use SVE for 64-bit vectors.
define i16 @sminv_v4i16(<4 x i16> %a) #0 {
; CHECK-LABEL: sminv_v4i16:
; CHECK: sminv h0, v0.4h
; CHECK: ret
  %res = call i16 @llvm.vector.reduce.smin.v4i16(<4 x i16> %a)
  ret i16 %res
}

; Don't use SVE for 128-bit vectors.
define i16 @sminv_v8i16(<8 x i16> %a) #0 {
; CHECK-LABEL: sminv_v8i16:
; CHECK: sminv h0, v0.8h
; CHECK: ret
  %res = call i16 @llvm.vector.reduce.smin.v8i16(<8 x i16> %a)
  ret i16 %res
}

define i16 @sminv_v16i16(<16 x i16>* %a) #0 {
; CHECK-LABEL: sminv_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-NEXT: sminv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; CHECK-NEXT: fmov w0, s[[REDUCE]]
; CHECK-NEXT: ret
  %op = load <16 x i16>, <16 x i16>* %a
  %res = call i16 @llvm.vector.reduce.smin.v16i16(<16 x i16> %op)
  ret i16 %res
}

define i16 @sminv_v32i16(<32 x i16>* %a) #0 {
; CHECK-LABEL: sminv_v32i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: sminv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_512-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #16
; VBITS_EQ_256-DAG: ld1h { [[LO:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[HI:z[0-9]+]].h }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #1]
; VBITS_EQ_256-DAG: smin [[MIN:z[0-9]+]].h, [[PG]]/m, [[HI]].h, [[LO]].h
; VBITS_EQ_256-DAG: sminv h[[REDUCE:[0-9]+]], [[PG]], [[MIN]].h
; VBITS_EQ_256-NEXT: fmov w0, s[[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <32 x i16>, <32 x i16>* %a
  %res = call i16 @llvm.vector.reduce.smin.v32i16(<32 x i16> %op)
  ret i16 %res
}

define i16 @sminv_v64i16(<64 x i16>* %a) #0 {
; CHECK-LABEL: sminv_v64i16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: sminv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_1024-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <64 x i16>, <64 x i16>* %a
  %res = call i16 @llvm.vector.reduce.smin.v64i16(<64 x i16> %op)
  ret i16 %res
}

define i16 @sminv_v128i16(<128 x i16>* %a) #0 {
; CHECK-LABEL: sminv_v128i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: sminv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_2048-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <128 x i16>, <128 x i16>* %a
  %res = call i16 @llvm.vector.reduce.smin.v128i16(<128 x i16> %op)
  ret i16 %res
}

; Don't use SVE for 64-bit vectors.
define i32 @sminv_v2i32(<2 x i32> %a) #0 {
; CHECK-LABEL: sminv_v2i32:
; CHECK: minp v0.2s, v0.2s
; CHECK: ret
  %res = call i32 @llvm.vector.reduce.smin.v2i32(<2 x i32> %a)
  ret i32 %res
}

; Don't use SVE for 128-bit vectors.
define i32 @sminv_v4i32(<4 x i32> %a) #0 {
; CHECK-LABEL: sminv_v4i32:
; CHECK: sminv s0, v0.4s
; CHECK: ret
  %res = call i32 @llvm.vector.reduce.smin.v4i32(<4 x i32> %a)
  ret i32 %res
}

define i32 @sminv_v8i32(<8 x i32>* %a) #0 {
; CHECK-LABEL: sminv_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: sminv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; CHECK-NEXT: fmov w0, [[REDUCE]]
; CHECK-NEXT: ret
  %op = load <8 x i32>, <8 x i32>* %a
  %res = call i32 @llvm.vector.reduce.smin.v8i32(<8 x i32> %op)
  ret i32 %res
}

define i32 @sminv_v16i32(<16 x i32>* %a) #0 {
; CHECK-LABEL: sminv_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: sminv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_512-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #8
; VBITS_EQ_256-DAG: ld1w { [[LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[HI:z[0-9]+]].s }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #2]
; VBITS_EQ_256-DAG: smin [[MIN:z[0-9]+]].s, [[PG]]/m, [[HI]].s, [[LO]].s
; VBITS_EQ_256-DAG: sminv [[REDUCE:s[0-9]+]], [[PG]], [[MIN]].s
; VBITS_EQ_256-NEXT: fmov w0, [[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <16 x i32>, <16 x i32>* %a
  %res = call i32 @llvm.vector.reduce.smin.v16i32(<16 x i32> %op)
  ret i32 %res
}

define i32 @sminv_v32i32(<32 x i32>* %a) #0 {
; CHECK-LABEL: sminv_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: sminv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_1024-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <32 x i32>, <32 x i32>* %a
  %res = call i32 @llvm.vector.reduce.smin.v32i32(<32 x i32> %op)
  ret i32 %res
}

define i32 @sminv_v64i32(<64 x i32>* %a) #0 {
; CHECK-LABEL: sminv_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: sminv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_2048-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <64 x i32>, <64 x i32>* %a
  %res = call i32 @llvm.vector.reduce.smin.v64i32(<64 x i32> %op)
  ret i32 %res
}

; Nothing to do for single element vectors.
define i64 @sminv_v1i64(<1 x i64> %a) #0 {
; CHECK-LABEL: sminv_v1i64:
; CHECK: fmov x0, d0
; CHECK: ret
  %res = call i64 @llvm.vector.reduce.smin.v1i64(<1 x i64> %a)
  ret i64 %res
}

; No NEON 64-bit vector SMINV support. Use SVE.
define i64 @sminv_v2i64(<2 x i64> %a) #0 {
; CHECK-LABEL: sminv_v2i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl2
; CHECK-NEXT: sminv [[REDUCE:d[0-9]+]], [[PG]], z0.d
; CHECK-NEXT: fmov x0, [[REDUCE]]
; CHECK-NEXT: ret
  %res = call i64 @llvm.vector.reduce.smin.v2i64(<2 x i64> %a)
  ret i64 %res
}

define i64 @sminv_v4i64(<4 x i64>* %a) #0 {
; CHECK-LABEL: sminv_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: sminv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; CHECK-NEXT: fmov x0, [[REDUCE]]
; CHECK-NEXT: ret
  %op = load <4 x i64>, <4 x i64>* %a
  %res = call i64 @llvm.vector.reduce.smin.v4i64(<4 x i64> %op)
  ret i64 %res
}

define i64 @sminv_v8i64(<8 x i64>* %a) #0 {
; CHECK-LABEL: sminv_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: sminv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_512-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: ld1d { [[LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[HI:z[0-9]+]].d }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-DAG: smin [[MIN:z[0-9]+]].d, [[PG]]/m, [[HI]].d, [[LO]].d
; VBITS_EQ_256-DAG: sminv [[REDUCE:d[0-9]+]], [[PG]], [[MIN]].d
; VBITS_EQ_256-NEXT: fmov x0, [[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <8 x i64>, <8 x i64>* %a
  %res = call i64 @llvm.vector.reduce.smin.v8i64(<8 x i64> %op)
  ret i64 %res
}

define i64 @sminv_v16i64(<16 x i64>* %a) #0 {
; CHECK-LABEL: sminv_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: sminv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_1024-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <16 x i64>, <16 x i64>* %a
  %res = call i64 @llvm.vector.reduce.smin.v16i64(<16 x i64> %op)
  ret i64 %res
}

define i64 @sminv_v32i64(<32 x i64>* %a) #0 {
; CHECK-LABEL: sminv_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: sminv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_2048-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <32 x i64>, <32 x i64>* %a
  %res = call i64 @llvm.vector.reduce.smin.v32i64(<32 x i64> %op)
  ret i64 %res
}

;
; UMAXV
;

; Don't use SVE for 64-bit vectors.
define i8 @umaxv_v8i8(<8 x i8> %a) #0 {
; CHECK-LABEL: umaxv_v8i8:
; CHECK: umaxv b0, v0.8b
; CHECK: ret
  %res = call i8 @llvm.vector.reduce.umax.v8i8(<8 x i8> %a)
  ret i8 %res
}

; Don't use SVE for 128-bit vectors.
define i8 @umaxv_v16i8(<16 x i8> %a) #0 {
; CHECK-LABEL: umaxv_v16i8:
; CHECK: umaxv b0, v0.16b
; CHECK: ret
  %res = call i8 @llvm.vector.reduce.umax.v16i8(<16 x i8> %a)
  ret i8 %res
}

define i8 @umaxv_v32i8(<32 x i8>* %a) #0 {
; CHECK-LABEL: umaxv_v32i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl32
; CHECK-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-NEXT: umaxv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; CHECK-NEXT: fmov w0, s[[REDUCE]]
; CHECK-NEXT: ret
  %op = load <32 x i8>, <32 x i8>* %a
  %res = call i8 @llvm.vector.reduce.umax.v32i8(<32 x i8> %op)
  ret i8 %res
}

define i8 @umaxv_v64i8(<64 x i8>* %a) #0 {
; CHECK-LABEL: umaxv_v64i8:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].b, vl64
; VBITS_GE_512-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: umaxv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_512-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].b, vl32
; VBITS_EQ_256-DAG: mov w[[NUMELTS:[0-9]+]], #32
; VBITS_EQ_256-DAG: ld1b { [[LO:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1b { [[HI:z[0-9]+]].b }, [[PG]]/z, [x0, x[[NUMELTS]]]
; VBITS_EQ_256-DAG: umax [[MAX:z[0-9]+]].b, [[PG]]/m, [[HI]].b, [[LO]].b
; VBITS_EQ_256-DAG: umaxv b[[REDUCE:[0-9]+]], [[PG]], [[MAX]].b
; VBITS_EQ_256-NEXT: fmov w0, s[[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <64 x i8>, <64 x i8>* %a
  %res = call i8 @llvm.vector.reduce.umax.v64i8(<64 x i8> %op)
  ret i8 %res
}

define i8 @umaxv_v128i8(<128 x i8>* %a) #0 {
; CHECK-LABEL: umaxv_v128i8:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].b, vl128
; VBITS_GE_1024-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: umaxv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_1024-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <128 x i8>, <128 x i8>* %a
  %res = call i8 @llvm.vector.reduce.umax.v128i8(<128 x i8> %op)
  ret i8 %res
}

define i8 @umaxv_v256i8(<256 x i8>* %a) #0 {
; CHECK-LABEL: umaxv_v256i8:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].b, vl256
; VBITS_GE_2048-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: umaxv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_2048-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <256 x i8>, <256 x i8>* %a
  %res = call i8 @llvm.vector.reduce.umax.v256i8(<256 x i8> %op)
  ret i8 %res
}

; Don't use SVE for 64-bit vectors.
define i16 @umaxv_v4i16(<4 x i16> %a) #0 {
; CHECK-LABEL: umaxv_v4i16:
; CHECK: umaxv h0, v0.4h
; CHECK: ret
  %res = call i16 @llvm.vector.reduce.umax.v4i16(<4 x i16> %a)
  ret i16 %res
}

; Don't use SVE for 128-bit vectors.
define i16 @umaxv_v8i16(<8 x i16> %a) #0 {
; CHECK-LABEL: umaxv_v8i16:
; CHECK: umaxv h0, v0.8h
; CHECK: ret
  %res = call i16 @llvm.vector.reduce.umax.v8i16(<8 x i16> %a)
  ret i16 %res
}

define i16 @umaxv_v16i16(<16 x i16>* %a) #0 {
; CHECK-LABEL: umaxv_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-NEXT: umaxv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; CHECK-NEXT: fmov w0, s[[REDUCE]]
; CHECK-NEXT: ret
  %op = load <16 x i16>, <16 x i16>* %a
  %res = call i16 @llvm.vector.reduce.umax.v16i16(<16 x i16> %op)
  ret i16 %res
}

define i16 @umaxv_v32i16(<32 x i16>* %a) #0 {
; CHECK-LABEL: umaxv_v32i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: umaxv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_512-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #16
; VBITS_EQ_256-DAG: ld1h { [[LO:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[HI:z[0-9]+]].h }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #1]
; VBITS_EQ_256-DAG: umax [[MAX:z[0-9]+]].h, [[PG]]/m, [[HI]].h, [[LO]].h
; VBITS_EQ_256-DAG: umaxv h[[REDUCE:[0-9]+]], [[PG]], [[MAX]].h
; VBITS_EQ_256-NEXT: fmov w0, s[[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <32 x i16>, <32 x i16>* %a
  %res = call i16 @llvm.vector.reduce.umax.v32i16(<32 x i16> %op)
  ret i16 %res
}

define i16 @umaxv_v64i16(<64 x i16>* %a) #0 {
; CHECK-LABEL: umaxv_v64i16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: umaxv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_1024-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <64 x i16>, <64 x i16>* %a
  %res = call i16 @llvm.vector.reduce.umax.v64i16(<64 x i16> %op)
  ret i16 %res
}

define i16 @umaxv_v128i16(<128 x i16>* %a) #0 {
; CHECK-LABEL: umaxv_v128i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: umaxv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_2048-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <128 x i16>, <128 x i16>* %a
  %res = call i16 @llvm.vector.reduce.umax.v128i16(<128 x i16> %op)
  ret i16 %res
}

; Don't use SVE for 64-bit vectors.
define i32 @umaxv_v2i32(<2 x i32> %a) #0 {
; CHECK-LABEL: umaxv_v2i32:
; CHECK: umaxp v0.2s, v0.2s
; CHECK: ret
  %res = call i32 @llvm.vector.reduce.umax.v2i32(<2 x i32> %a)
  ret i32 %res
}

; Don't use SVE for 128-bit vectors.
define i32 @umaxv_v4i32(<4 x i32> %a) #0 {
; CHECK-LABEL: umaxv_v4i32:
; CHECK: umaxv s0, v0.4s
; CHECK: ret
  %res = call i32 @llvm.vector.reduce.umax.v4i32(<4 x i32> %a)
  ret i32 %res
}

define i32 @umaxv_v8i32(<8 x i32>* %a) #0 {
; CHECK-LABEL: umaxv_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: umaxv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; CHECK-NEXT: fmov w0, [[REDUCE]]
; CHECK-NEXT: ret
  %op = load <8 x i32>, <8 x i32>* %a
  %res = call i32 @llvm.vector.reduce.umax.v8i32(<8 x i32> %op)
  ret i32 %res
}

define i32 @umaxv_v16i32(<16 x i32>* %a) #0 {
; CHECK-LABEL: umaxv_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: umaxv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_512-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #8
; VBITS_EQ_256-DAG: ld1w { [[LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[HI:z[0-9]+]].s }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #2]
; VBITS_EQ_256-DAG: umax [[MAX:z[0-9]+]].s, [[PG]]/m, [[HI]].s, [[LO]].s
; VBITS_EQ_256-DAG: umaxv [[REDUCE:s[0-9]+]], [[PG]], [[MAX]].s
; VBITS_EQ_256-NEXT: fmov w0, [[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <16 x i32>, <16 x i32>* %a
  %res = call i32 @llvm.vector.reduce.umax.v16i32(<16 x i32> %op)
  ret i32 %res
}

define i32 @umaxv_v32i32(<32 x i32>* %a) #0 {
; CHECK-LABEL: umaxv_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: umaxv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_1024-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <32 x i32>, <32 x i32>* %a
  %res = call i32 @llvm.vector.reduce.umax.v32i32(<32 x i32> %op)
  ret i32 %res
}

define i32 @umaxv_v64i32(<64 x i32>* %a) #0 {
; CHECK-LABEL: umaxv_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: umaxv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_2048-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <64 x i32>, <64 x i32>* %a
  %res = call i32 @llvm.vector.reduce.umax.v64i32(<64 x i32> %op)
  ret i32 %res
}

; Nothing to do for single element vectors.
define i64 @umaxv_v1i64(<1 x i64> %a) #0 {
; CHECK-LABEL: umaxv_v1i64:
; CHECK: fmov x0, d0
; CHECK: ret
  %res = call i64 @llvm.vector.reduce.umax.v1i64(<1 x i64> %a)
  ret i64 %res
}

; No NEON 64-bit vector UMAXV support. Use SVE.
define i64 @umaxv_v2i64(<2 x i64> %a) #0 {
; CHECK-LABEL: umaxv_v2i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl2
; CHECK-NEXT: umaxv [[REDUCE:d[0-9]+]], [[PG]], z0.d
; CHECK-NEXT: fmov x0, [[REDUCE]]
; CHECK-NEXT: ret
  %res = call i64 @llvm.vector.reduce.umax.v2i64(<2 x i64> %a)
  ret i64 %res
}

define i64 @umaxv_v4i64(<4 x i64>* %a) #0 {
; CHECK-LABEL: umaxv_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: umaxv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; CHECK-NEXT: fmov x0, [[REDUCE]]
; CHECK-NEXT: ret
  %op = load <4 x i64>, <4 x i64>* %a
  %res = call i64 @llvm.vector.reduce.umax.v4i64(<4 x i64> %op)
  ret i64 %res
}

define i64 @umaxv_v8i64(<8 x i64>* %a) #0 {
; CHECK-LABEL: umaxv_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: umaxv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_512-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: ld1d { [[LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[HI:z[0-9]+]].d }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-DAG: umax [[MAX:z[0-9]+]].d, [[PG]]/m, [[HI]].d, [[LO]].d
; VBITS_EQ_256-DAG: umaxv [[REDUCE:d[0-9]+]], [[PG]], [[MAX]].d
; VBITS_EQ_256-NEXT: fmov x0, [[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <8 x i64>, <8 x i64>* %a
  %res = call i64 @llvm.vector.reduce.umax.v8i64(<8 x i64> %op)
  ret i64 %res
}

define i64 @umaxv_v16i64(<16 x i64>* %a) #0 {
; CHECK-LABEL: umaxv_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: umaxv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_1024-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <16 x i64>, <16 x i64>* %a
  %res = call i64 @llvm.vector.reduce.umax.v16i64(<16 x i64> %op)
  ret i64 %res
}

define i64 @umaxv_v32i64(<32 x i64>* %a) #0 {
; CHECK-LABEL: umaxv_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: umaxv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_2048-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <32 x i64>, <32 x i64>* %a
  %res = call i64 @llvm.vector.reduce.umax.v32i64(<32 x i64> %op)
  ret i64 %res
}

;
; UMINV
;

; Don't use SVE for 64-bit vectors.
define i8 @uminv_v8i8(<8 x i8> %a) #0 {
; CHECK-LABEL: uminv_v8i8:
; CHECK: uminv b0, v0.8b
; CHECK: ret
  %res = call i8 @llvm.vector.reduce.umin.v8i8(<8 x i8> %a)
  ret i8 %res
}

; Don't use SVE for 128-bit vectors.
define i8 @uminv_v16i8(<16 x i8> %a) #0 {
; CHECK-LABEL: uminv_v16i8:
; CHECK: uminv b0, v0.16b
; CHECK: ret
  %res = call i8 @llvm.vector.reduce.umin.v16i8(<16 x i8> %a)
  ret i8 %res
}

define i8 @uminv_v32i8(<32 x i8>* %a) #0 {
; CHECK-LABEL: uminv_v32i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl32
; CHECK-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-NEXT: uminv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; CHECK-NEXT: fmov w0, s[[REDUCE]]
; CHECK-NEXT: ret
  %op = load <32 x i8>, <32 x i8>* %a
  %res = call i8 @llvm.vector.reduce.umin.v32i8(<32 x i8> %op)
  ret i8 %res
}

define i8 @uminv_v64i8(<64 x i8>* %a) #0 {
; CHECK-LABEL: uminv_v64i8:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].b, vl64
; VBITS_GE_512-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: uminv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_512-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].b, vl32
; VBITS_EQ_256-DAG: mov w[[NUMELTS:[0-9]+]], #32
; VBITS_EQ_256-DAG: ld1b { [[LO:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1b { [[HI:z[0-9]+]].b }, [[PG]]/z, [x0, x[[NUMELTS]]]
; VBITS_EQ_256-DAG: umin [[MIN:z[0-9]+]].b, [[PG]]/m, [[HI]].b, [[LO]].b
; VBITS_EQ_256-DAG: uminv b[[REDUCE:[0-9]+]], [[PG]], [[MIN]].b
; VBITS_EQ_256-NEXT: fmov w0, s[[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <64 x i8>, <64 x i8>* %a
  %res = call i8 @llvm.vector.reduce.umin.v64i8(<64 x i8> %op)
  ret i8 %res
}

define i8 @uminv_v128i8(<128 x i8>* %a) #0 {
; CHECK-LABEL: uminv_v128i8:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].b, vl128
; VBITS_GE_1024-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: uminv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_1024-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <128 x i8>, <128 x i8>* %a
  %res = call i8 @llvm.vector.reduce.umin.v128i8(<128 x i8> %op)
  ret i8 %res
}

define i8 @uminv_v256i8(<256 x i8>* %a) #0 {
; CHECK-LABEL: uminv_v256i8:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].b, vl256
; VBITS_GE_2048-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: uminv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_2048-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <256 x i8>, <256 x i8>* %a
  %res = call i8 @llvm.vector.reduce.umin.v256i8(<256 x i8> %op)
  ret i8 %res
}

; Don't use SVE for 64-bit vectors.
define i16 @uminv_v4i16(<4 x i16> %a) #0 {
; CHECK-LABEL: uminv_v4i16:
; CHECK: uminv h0, v0.4h
; CHECK: ret
  %res = call i16 @llvm.vector.reduce.umin.v4i16(<4 x i16> %a)
  ret i16 %res
}

; Don't use SVE for 128-bit vectors.
define i16 @uminv_v8i16(<8 x i16> %a) #0 {
; CHECK-LABEL: uminv_v8i16:
; CHECK: uminv h0, v0.8h
; CHECK: ret
  %res = call i16 @llvm.vector.reduce.umin.v8i16(<8 x i16> %a)
  ret i16 %res
}

define i16 @uminv_v16i16(<16 x i16>* %a) #0 {
; CHECK-LABEL: uminv_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-NEXT: uminv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; CHECK-NEXT: fmov w0, s[[REDUCE]]
; CHECK-NEXT: ret
  %op = load <16 x i16>, <16 x i16>* %a
  %res = call i16 @llvm.vector.reduce.umin.v16i16(<16 x i16> %op)
  ret i16 %res
}

define i16 @uminv_v32i16(<32 x i16>* %a) #0 {
; CHECK-LABEL: uminv_v32i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: uminv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_512-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #16
; VBITS_EQ_256-DAG: ld1h { [[LO:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[HI:z[0-9]+]].h }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #1]
; VBITS_EQ_256-DAG: umin [[MIN:z[0-9]+]].h, [[PG]]/m, [[HI]].h, [[LO]].h
; VBITS_EQ_256-DAG: uminv h[[REDUCE:[0-9]+]], [[PG]], [[MIN]].h
; VBITS_EQ_256-NEXT: fmov w0, s[[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <32 x i16>, <32 x i16>* %a
  %res = call i16 @llvm.vector.reduce.umin.v32i16(<32 x i16> %op)
  ret i16 %res
}

define i16 @uminv_v64i16(<64 x i16>* %a) #0 {
; CHECK-LABEL: uminv_v64i16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: uminv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_1024-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <64 x i16>, <64 x i16>* %a
  %res = call i16 @llvm.vector.reduce.umin.v64i16(<64 x i16> %op)
  ret i16 %res
}

define i16 @uminv_v128i16(<128 x i16>* %a) #0 {
; CHECK-LABEL: uminv_v128i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: uminv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_2048-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <128 x i16>, <128 x i16>* %a
  %res = call i16 @llvm.vector.reduce.umin.v128i16(<128 x i16> %op)
  ret i16 %res
}

; Don't use SVE for 64-bit vectors.
define i32 @uminv_v2i32(<2 x i32> %a) #0 {
; CHECK-LABEL: uminv_v2i32:
; CHECK: minp v0.2s, v0.2s
; CHECK: ret
  %res = call i32 @llvm.vector.reduce.umin.v2i32(<2 x i32> %a)
  ret i32 %res
}

; Don't use SVE for 128-bit vectors.
define i32 @uminv_v4i32(<4 x i32> %a) #0 {
; CHECK-LABEL: uminv_v4i32:
; CHECK: uminv s0, v0.4s
; CHECK: ret
  %res = call i32 @llvm.vector.reduce.umin.v4i32(<4 x i32> %a)
  ret i32 %res
}

define i32 @uminv_v8i32(<8 x i32>* %a) #0 {
; CHECK-LABEL: uminv_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: uminv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; CHECK-NEXT: fmov w0, [[REDUCE]]
; CHECK-NEXT: ret
  %op = load <8 x i32>, <8 x i32>* %a
  %res = call i32 @llvm.vector.reduce.umin.v8i32(<8 x i32> %op)
  ret i32 %res
}

define i32 @uminv_v16i32(<16 x i32>* %a) #0 {
; CHECK-LABEL: uminv_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: uminv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_512-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #8
; VBITS_EQ_256-DAG: ld1w { [[LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[HI:z[0-9]+]].s }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #2]
; VBITS_EQ_256-DAG: umin [[MIN:z[0-9]+]].s, [[PG]]/m, [[HI]].s, [[LO]].s
; VBITS_EQ_256-DAG: uminv [[REDUCE:s[0-9]+]], [[PG]], [[MIN]].s
; VBITS_EQ_256-NEXT: fmov w0, [[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <16 x i32>, <16 x i32>* %a
  %res = call i32 @llvm.vector.reduce.umin.v16i32(<16 x i32> %op)
  ret i32 %res
}

define i32 @uminv_v32i32(<32 x i32>* %a) #0 {
; CHECK-LABEL: uminv_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: uminv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_1024-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <32 x i32>, <32 x i32>* %a
  %res = call i32 @llvm.vector.reduce.umin.v32i32(<32 x i32> %op)
  ret i32 %res
}

define i32 @uminv_v64i32(<64 x i32>* %a) #0 {
; CHECK-LABEL: uminv_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: uminv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_2048-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <64 x i32>, <64 x i32>* %a
  %res = call i32 @llvm.vector.reduce.umin.v64i32(<64 x i32> %op)
  ret i32 %res
}

; Nothing to do for single element vectors.
define i64 @uminv_v1i64(<1 x i64> %a) #0 {
; CHECK-LABEL: uminv_v1i64:
; CHECK: fmov x0, d0
; CHECK: ret
  %res = call i64 @llvm.vector.reduce.umin.v1i64(<1 x i64> %a)
  ret i64 %res
}

; No NEON 64-bit vector UMINV support. Use SVE.
define i64 @uminv_v2i64(<2 x i64> %a) #0 {
; CHECK-LABEL: uminv_v2i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl2
; CHECK-NEXT: uminv [[REDUCE:d[0-9]+]], [[PG]], z0.d
; CHECK-NEXT: fmov x0, [[REDUCE]]
; CHECK-NEXT: ret
  %res = call i64 @llvm.vector.reduce.umin.v2i64(<2 x i64> %a)
  ret i64 %res
}

define i64 @uminv_v4i64(<4 x i64>* %a) #0 {
; CHECK-LABEL: uminv_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: uminv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; CHECK-NEXT: fmov x0, [[REDUCE]]
; CHECK-NEXT: ret
  %op = load <4 x i64>, <4 x i64>* %a
  %res = call i64 @llvm.vector.reduce.umin.v4i64(<4 x i64> %op)
  ret i64 %res
}

define i64 @uminv_v8i64(<8 x i64>* %a) #0 {
; CHECK-LABEL: uminv_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: uminv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_512-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: ld1d { [[LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[HI:z[0-9]+]].d }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-DAG: umin [[MIN:z[0-9]+]].d, [[PG]]/m, [[HI]].d, [[LO]].d
; VBITS_EQ_256-DAG: uminv [[REDUCE:d[0-9]+]], [[PG]], [[MIN]].d
; VBITS_EQ_256-NEXT: fmov x0, [[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <8 x i64>, <8 x i64>* %a
  %res = call i64 @llvm.vector.reduce.umin.v8i64(<8 x i64> %op)
  ret i64 %res
}

define i64 @uminv_v16i64(<16 x i64>* %a) #0 {
; CHECK-LABEL: uminv_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: uminv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_1024-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <16 x i64>, <16 x i64>* %a
  %res = call i64 @llvm.vector.reduce.umin.v16i64(<16 x i64> %op)
  ret i64 %res
}

define i64 @uminv_v32i64(<32 x i64>* %a) #0 {
; CHECK-LABEL: uminv_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: uminv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_2048-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <32 x i64>, <32 x i64>* %a
  %res = call i64 @llvm.vector.reduce.umin.v32i64(<32 x i64> %op)
  ret i64 %res
}

attributes #0 = { "target-features"="+sve" }

declare i8 @llvm.vector.reduce.add.v8i8(<8 x i8>)
declare i8 @llvm.vector.reduce.add.v16i8(<16 x i8>)
declare i8 @llvm.vector.reduce.add.v32i8(<32 x i8>)
declare i8 @llvm.vector.reduce.add.v64i8(<64 x i8>)
declare i8 @llvm.vector.reduce.add.v128i8(<128 x i8>)
declare i8 @llvm.vector.reduce.add.v256i8(<256 x i8>)

declare i16 @llvm.vector.reduce.add.v4i16(<4 x i16>)
declare i16 @llvm.vector.reduce.add.v8i16(<8 x i16>)
declare i16 @llvm.vector.reduce.add.v16i16(<16 x i16>)
declare i16 @llvm.vector.reduce.add.v32i16(<32 x i16>)
declare i16 @llvm.vector.reduce.add.v64i16(<64 x i16>)
declare i16 @llvm.vector.reduce.add.v128i16(<128 x i16>)

declare i32 @llvm.vector.reduce.add.v2i32(<2 x i32>)
declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>)
declare i32 @llvm.vector.reduce.add.v8i32(<8 x i32>)
declare i32 @llvm.vector.reduce.add.v16i32(<16 x i32>)
declare i32 @llvm.vector.reduce.add.v32i32(<32 x i32>)
declare i32 @llvm.vector.reduce.add.v64i32(<64 x i32>)

declare i64 @llvm.vector.reduce.add.v1i64(<1 x i64>)
declare i64 @llvm.vector.reduce.add.v2i64(<2 x i64>)
declare i64 @llvm.vector.reduce.add.v4i64(<4 x i64>)
declare i64 @llvm.vector.reduce.add.v8i64(<8 x i64>)
declare i64 @llvm.vector.reduce.add.v16i64(<16 x i64>)
declare i64 @llvm.vector.reduce.add.v32i64(<32 x i64>)

declare i8 @llvm.vector.reduce.smax.v8i8(<8 x i8>)
declare i8 @llvm.vector.reduce.smax.v16i8(<16 x i8>)
declare i8 @llvm.vector.reduce.smax.v32i8(<32 x i8>)
declare i8 @llvm.vector.reduce.smax.v64i8(<64 x i8>)
declare i8 @llvm.vector.reduce.smax.v128i8(<128 x i8>)
declare i8 @llvm.vector.reduce.smax.v256i8(<256 x i8>)

declare i16 @llvm.vector.reduce.smax.v4i16(<4 x i16>)
declare i16 @llvm.vector.reduce.smax.v8i16(<8 x i16>)
declare i16 @llvm.vector.reduce.smax.v16i16(<16 x i16>)
declare i16 @llvm.vector.reduce.smax.v32i16(<32 x i16>)
declare i16 @llvm.vector.reduce.smax.v64i16(<64 x i16>)
declare i16 @llvm.vector.reduce.smax.v128i16(<128 x i16>)

declare i32 @llvm.vector.reduce.smax.v2i32(<2 x i32>)
declare i32 @llvm.vector.reduce.smax.v4i32(<4 x i32>)
declare i32 @llvm.vector.reduce.smax.v8i32(<8 x i32>)
declare i32 @llvm.vector.reduce.smax.v16i32(<16 x i32>)
declare i32 @llvm.vector.reduce.smax.v32i32(<32 x i32>)
declare i32 @llvm.vector.reduce.smax.v64i32(<64 x i32>)

declare i64 @llvm.vector.reduce.smax.v1i64(<1 x i64>)
declare i64 @llvm.vector.reduce.smax.v2i64(<2 x i64>)
declare i64 @llvm.vector.reduce.smax.v4i64(<4 x i64>)
declare i64 @llvm.vector.reduce.smax.v8i64(<8 x i64>)
declare i64 @llvm.vector.reduce.smax.v16i64(<16 x i64>)
declare i64 @llvm.vector.reduce.smax.v32i64(<32 x i64>)

declare i8 @llvm.vector.reduce.smin.v8i8(<8 x i8>)
declare i8 @llvm.vector.reduce.smin.v16i8(<16 x i8>)
declare i8 @llvm.vector.reduce.smin.v32i8(<32 x i8>)
declare i8 @llvm.vector.reduce.smin.v64i8(<64 x i8>)
declare i8 @llvm.vector.reduce.smin.v128i8(<128 x i8>)
declare i8 @llvm.vector.reduce.smin.v256i8(<256 x i8>)

declare i16 @llvm.vector.reduce.smin.v4i16(<4 x i16>)
declare i16 @llvm.vector.reduce.smin.v8i16(<8 x i16>)
declare i16 @llvm.vector.reduce.smin.v16i16(<16 x i16>)
declare i16 @llvm.vector.reduce.smin.v32i16(<32 x i16>)
declare i16 @llvm.vector.reduce.smin.v64i16(<64 x i16>)
declare i16 @llvm.vector.reduce.smin.v128i16(<128 x i16>)

declare i32 @llvm.vector.reduce.smin.v2i32(<2 x i32>)
declare i32 @llvm.vector.reduce.smin.v4i32(<4 x i32>)
declare i32 @llvm.vector.reduce.smin.v8i32(<8 x i32>)
declare i32 @llvm.vector.reduce.smin.v16i32(<16 x i32>)
declare i32 @llvm.vector.reduce.smin.v32i32(<32 x i32>)
declare i32 @llvm.vector.reduce.smin.v64i32(<64 x i32>)

declare i64 @llvm.vector.reduce.smin.v1i64(<1 x i64>)
declare i64 @llvm.vector.reduce.smin.v2i64(<2 x i64>)
declare i64 @llvm.vector.reduce.smin.v4i64(<4 x i64>)
declare i64 @llvm.vector.reduce.smin.v8i64(<8 x i64>)
declare i64 @llvm.vector.reduce.smin.v16i64(<16 x i64>)
declare i64 @llvm.vector.reduce.smin.v32i64(<32 x i64>)

declare i8 @llvm.vector.reduce.umax.v8i8(<8 x i8>)
declare i8 @llvm.vector.reduce.umax.v16i8(<16 x i8>)
declare i8 @llvm.vector.reduce.umax.v32i8(<32 x i8>)
declare i8 @llvm.vector.reduce.umax.v64i8(<64 x i8>)
declare i8 @llvm.vector.reduce.umax.v128i8(<128 x i8>)
declare i8 @llvm.vector.reduce.umax.v256i8(<256 x i8>)

declare i16 @llvm.vector.reduce.umax.v4i16(<4 x i16>)
declare i16 @llvm.vector.reduce.umax.v8i16(<8 x i16>)
declare i16 @llvm.vector.reduce.umax.v16i16(<16 x i16>)
declare i16 @llvm.vector.reduce.umax.v32i16(<32 x i16>)
declare i16 @llvm.vector.reduce.umax.v64i16(<64 x i16>)
declare i16 @llvm.vector.reduce.umax.v128i16(<128 x i16>)

declare i32 @llvm.vector.reduce.umax.v2i32(<2 x i32>)
declare i32 @llvm.vector.reduce.umax.v4i32(<4 x i32>)
declare i32 @llvm.vector.reduce.umax.v8i32(<8 x i32>)
declare i32 @llvm.vector.reduce.umax.v16i32(<16 x i32>)
declare i32 @llvm.vector.reduce.umax.v32i32(<32 x i32>)
declare i32 @llvm.vector.reduce.umax.v64i32(<64 x i32>)

declare i64 @llvm.vector.reduce.umax.v1i64(<1 x i64>)
declare i64 @llvm.vector.reduce.umax.v2i64(<2 x i64>)
declare i64 @llvm.vector.reduce.umax.v4i64(<4 x i64>)
declare i64 @llvm.vector.reduce.umax.v8i64(<8 x i64>)
declare i64 @llvm.vector.reduce.umax.v16i64(<16 x i64>)
declare i64 @llvm.vector.reduce.umax.v32i64(<32 x i64>)

declare i8 @llvm.vector.reduce.umin.v8i8(<8 x i8>)
declare i8 @llvm.vector.reduce.umin.v16i8(<16 x i8>)
declare i8 @llvm.vector.reduce.umin.v32i8(<32 x i8>)
declare i8 @llvm.vector.reduce.umin.v64i8(<64 x i8>)
declare i8 @llvm.vector.reduce.umin.v128i8(<128 x i8>)
declare i8 @llvm.vector.reduce.umin.v256i8(<256 x i8>)

declare i16 @llvm.vector.reduce.umin.v4i16(<4 x i16>)
declare i16 @llvm.vector.reduce.umin.v8i16(<8 x i16>)
declare i16 @llvm.vector.reduce.umin.v16i16(<16 x i16>)
declare i16 @llvm.vector.reduce.umin.v32i16(<32 x i16>)
declare i16 @llvm.vector.reduce.umin.v64i16(<64 x i16>)
declare i16 @llvm.vector.reduce.umin.v128i16(<128 x i16>)

declare i32 @llvm.vector.reduce.umin.v2i32(<2 x i32>)
declare i32 @llvm.vector.reduce.umin.v4i32(<4 x i32>)
declare i32 @llvm.vector.reduce.umin.v8i32(<8 x i32>)
declare i32 @llvm.vector.reduce.umin.v16i32(<16 x i32>)
declare i32 @llvm.vector.reduce.umin.v32i32(<32 x i32>)
declare i32 @llvm.vector.reduce.umin.v64i32(<64 x i32>)

declare i64 @llvm.vector.reduce.umin.v1i64(<1 x i64>)
declare i64 @llvm.vector.reduce.umin.v2i64(<2 x i64>)
declare i64 @llvm.vector.reduce.umin.v4i64(<4 x i64>)
declare i64 @llvm.vector.reduce.umin.v8i64(<8 x i64>)
declare i64 @llvm.vector.reduce.umin.v16i64(<16 x i64>)
declare i64 @llvm.vector.reduce.umin.v32i64(<32 x i64>)
