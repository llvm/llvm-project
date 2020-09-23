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
  %res = call i8 @llvm.experimental.vector.reduce.add.v8i8(<8 x i8> %a)
  ret i8 %res
}

; Don't use SVE for 128-bit vectors.
define i8 @uaddv_v16i8(<16 x i8> %a) #0 {
; CHECK-LABEL: uaddv_v16i8:
; CHECK: addv b0, v0.16b
; CHECK: ret
  %res = call i8 @llvm.experimental.vector.reduce.add.v16i8(<16 x i8> %a)
  ret i8 %res
}

define i8 @uaddv_v32i8(<32 x i8>* %a) #0 {
; CHECK-LABEL: uaddv_v32i8:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].b, vl32
; VBITS_GE_256-DAG: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_256-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_256-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_256-NEXT: ret
  %op = load <32 x i8>, <32 x i8>* %a
  %res = call i8 @llvm.experimental.vector.reduce.add.v32i8(<32 x i8> %op)
  ret i8 %res
}

define i8 @uaddv_v64i8(<64 x i8>* %a) #0 {
; CHECK-LABEL: uaddv_v64i8:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].b, vl64
; VBITS_GE_512-DAG: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_512-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_512-NEXT: ret
  %op = load <64 x i8>, <64 x i8>* %a
  %res = call i8 @llvm.experimental.vector.reduce.add.v64i8(<64 x i8> %op)
  ret i8 %res
}

define i8 @uaddv_v128i8(<128 x i8>* %a) #0 {
; CHECK-LABEL: uaddv_v128i8:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].b, vl128
; VBITS_GE_1024-DAG: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_1024-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <128 x i8>, <128 x i8>* %a
  %res = call i8 @llvm.experimental.vector.reduce.add.v128i8(<128 x i8> %op)
  ret i8 %res
}

define i8 @uaddv_v256i8(<256 x i8>* %a) #0 {
; CHECK-LABEL: uaddv_v256i8:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].b, vl256
; VBITS_GE_2048-DAG: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_2048-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <256 x i8>, <256 x i8>* %a
  %res = call i8 @llvm.experimental.vector.reduce.add.v256i8(<256 x i8> %op)
  ret i8 %res
}

; Don't use SVE for 64-bit vectors.
define i16 @uaddv_v4i16(<4 x i16> %a) #0 {
; CHECK-LABEL: uaddv_v4i16:
; CHECK: addv h0, v0.4h
; CHECK: ret
  %res = call i16 @llvm.experimental.vector.reduce.add.v4i16(<4 x i16> %a)
  ret i16 %res
}

; Don't use SVE for 128-bit vectors.
define i16 @uaddv_v8i16(<8 x i16> %a) #0 {
; CHECK-LABEL: uaddv_v8i16:
; CHECK: addv h0, v0.8h
; CHECK: ret
  %res = call i16 @llvm.experimental.vector.reduce.add.v8i16(<8 x i16> %a)
  ret i16 %res
}

define i16 @uaddv_v16i16(<16 x i16>* %a) #0 {
; CHECK-LABEL: uaddv_v16i16:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_GE_256-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_256-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_256-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_256-NEXT: ret
  %op = load <16 x i16>, <16 x i16>* %a
  %res = call i16 @llvm.experimental.vector.reduce.add.v16i16(<16 x i16> %op)
  ret i16 %res
}

define i16 @uaddv_v32i16(<32 x i16>* %a) #0 {
; CHECK-LABEL: uaddv_v32i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_512-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_512-NEXT: ret
  %op = load <32 x i16>, <32 x i16>* %a
  %res = call i16 @llvm.experimental.vector.reduce.add.v32i16(<32 x i16> %op)
  ret i16 %res
}

define i16 @uaddv_v64i16(<64 x i16>* %a) #0 {
; CHECK-LABEL: uaddv_v64i16:
; VBITS_GE_1048: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1048-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1048-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_1048-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_1048-NEXT: ret
  %op = load <64 x i16>, <64 x i16>* %a
  %res = call i16 @llvm.experimental.vector.reduce.add.v64i16(<64 x i16> %op)
  ret i16 %res
}

define i16 @uaddv_v128i16(<128 x i16>* %a) #0 {
; CHECK-LABEL: uaddv_v128i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_2048-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <128 x i16>, <128 x i16>* %a
  %res = call i16 @llvm.experimental.vector.reduce.add.v128i16(<128 x i16> %op)
  ret i16 %res
}

; Don't use SVE for 64-bit vectors.
define i32 @uaddv_v2i32(<2 x i32> %a) #0 {
; CHECK-LABEL: uaddv_v2i32:
; CHECK: addp v0.2s, v0.2s
; CHECK: ret
  %res = call i32 @llvm.experimental.vector.reduce.add.v2i32(<2 x i32> %a)
  ret i32 %res
}

; Don't use SVE for 128-bit vectors.
define i32 @uaddv_v4i32(<4 x i32> %a) #0 {
; CHECK-LABEL: uaddv_v4i32:
; CHECK: addv s0, v0.4s
; CHECK: ret
  %res = call i32 @llvm.experimental.vector.reduce.add.v4i32(<4 x i32> %a)
  ret i32 %res
}

define i32 @uaddv_v8i32(<8 x i32>* %a) #0 {
; CHECK-LABEL: uaddv_v8i32:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_GE_256-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_256-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_256-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_256-NEXT: ret
  %op = load <8 x i32>, <8 x i32>* %a
  %res = call i32 @llvm.experimental.vector.reduce.add.v8i32(<8 x i32> %op)
  ret i32 %res
}

define i32 @uaddv_v16i32(<16 x i32>* %a) #0 {
; CHECK-LABEL: uaddv_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_512-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_512-NEXT: ret
  %op = load <16 x i32>, <16 x i32>* %a
  %res = call i32 @llvm.experimental.vector.reduce.add.v16i32(<16 x i32> %op)
  ret i32 %res
}

define i32 @uaddv_v32i32(<32 x i32>* %a) #0 {
; CHECK-LABEL: uaddv_v32i32:
; VBITS_GE_1048: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1048-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1048-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_1048-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_1048-NEXT: ret
  %op = load <32 x i32>, <32 x i32>* %a
  %res = call i32 @llvm.experimental.vector.reduce.add.v32i32(<32 x i32> %op)
  ret i32 %res
}

define i32 @uaddv_v64i32(<64 x i32>* %a) #0 {
; CHECK-LABEL: uaddv_v64i32:
; VBITS_GE_2096: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2096-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2096-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_2086-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_2096-NEXT: ret
  %op = load <64 x i32>, <64 x i32>* %a
  %res = call i32 @llvm.experimental.vector.reduce.add.v64i32(<64 x i32> %op)
  ret i32 %res
}

; Nothing to do for 64-bit vectors..
define i64 @uaddv_v1i64(<1 x i64> %a) #0 {
; CHECK-LABEL: uaddv_v1i64:
; CHECK: fmov x0, d0
; CHECK: ret
  %res = call i64 @llvm.experimental.vector.reduce.add.v1i64(<1 x i64> %a)
  ret i64 %res
}

; Don't use SVE for 128-bit vectors.
define i64 @uaddv_v2i64(<2 x i64> %a) #0 {
; CHECK-LABEL: uaddv_v2i64:
; CHECK: addp d0, v0.2d
; CHECK: ret
  %res = call i64 @llvm.experimental.vector.reduce.add.v2i64(<2 x i64> %a)
  ret i64 %res
}

define i64 @uaddv_v4i64(<4 x i64>* %a) #0 {
; CHECK-LABEL: uaddv_v4i64:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_GE_256-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_256-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_256-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_256-NEXT: ret
  %op = load <4 x i64>, <4 x i64>* %a
  %res = call i64 @llvm.experimental.vector.reduce.add.v4i64(<4 x i64> %op)
  ret i64 %res
}

define i64 @uaddv_v8i64(<8 x i64>* %a) #0 {
; CHECK-LABEL: uaddv_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_512-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_512-NEXT: ret
  %op = load <8 x i64>, <8 x i64>* %a
  %res = call i64 @llvm.experimental.vector.reduce.add.v8i64(<8 x i64> %op)
  ret i64 %res
}

define i64 @uaddv_v16i64(<16 x i64>* %a) #0 {
; CHECK-LABEL: uaddv_v16i64:
; VBITS_GE_1048: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1048-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1048-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_1048-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_1048-NEXT: ret
  %op = load <16 x i64>, <16 x i64>* %a
  %res = call i64 @llvm.experimental.vector.reduce.add.v16i64(<16 x i64> %op)
  ret i64 %res
}

define i64 @uaddv_v32i64(<32 x i64>* %a) #0 {
; CHECK-LABEL: uaddv_v32i64:
; VBITS_GE_2096: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2096-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2096-NEXT: uaddv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_2096-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_2096-NEXT: ret
  %op = load <32 x i64>, <32 x i64>* %a
  %res = call i64 @llvm.experimental.vector.reduce.add.v32i64(<32 x i64> %op)
  ret i64 %res
}

attributes #0 = { "target-features"="+sve" }

declare i8 @llvm.experimental.vector.reduce.add.v8i8(<8 x i8>)
declare i8 @llvm.experimental.vector.reduce.add.v16i8(<16 x i8>)
declare i8 @llvm.experimental.vector.reduce.add.v32i8(<32 x i8>)
declare i8 @llvm.experimental.vector.reduce.add.v64i8(<64 x i8>)
declare i8 @llvm.experimental.vector.reduce.add.v128i8(<128 x i8>)
declare i8 @llvm.experimental.vector.reduce.add.v256i8(<256 x i8>)

declare i16 @llvm.experimental.vector.reduce.add.v4i16(<4 x i16>)
declare i16 @llvm.experimental.vector.reduce.add.v8i16(<8 x i16>)
declare i16 @llvm.experimental.vector.reduce.add.v16i16(<16 x i16>)
declare i16 @llvm.experimental.vector.reduce.add.v32i16(<32 x i16>)
declare i16 @llvm.experimental.vector.reduce.add.v64i16(<64 x i16>)
declare i16 @llvm.experimental.vector.reduce.add.v128i16(<128 x i16>)

declare i32 @llvm.experimental.vector.reduce.add.v2i32(<2 x i32>)
declare i32 @llvm.experimental.vector.reduce.add.v4i32(<4 x i32>)
declare i32 @llvm.experimental.vector.reduce.add.v8i32(<8 x i32>)
declare i32 @llvm.experimental.vector.reduce.add.v16i32(<16 x i32>)
declare i32 @llvm.experimental.vector.reduce.add.v32i32(<32 x i32>)
declare i32 @llvm.experimental.vector.reduce.add.v64i32(<64 x i32>)

declare i64 @llvm.experimental.vector.reduce.add.v1i64(<1 x i64>)
declare i64 @llvm.experimental.vector.reduce.add.v2i64(<2 x i64>)
declare i64 @llvm.experimental.vector.reduce.add.v4i64(<4 x i64>)
declare i64 @llvm.experimental.vector.reduce.add.v8i64(<8 x i64>)
declare i64 @llvm.experimental.vector.reduce.add.v16i64(<16 x i64>)
declare i64 @llvm.experimental.vector.reduce.add.v32i64(<32 x i64>)
