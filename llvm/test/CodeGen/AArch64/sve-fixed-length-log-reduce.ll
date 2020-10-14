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
; ANDV
;

; No single instruction NEON ANDV support. Use SVE.
define i8 @andv_v8i8(<8 x i8> %a) #0 {
; CHECK-LABEL: andv_v8i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl8
; CHECK: andv b[[REDUCE:[0-9]+]], [[PG]], z0.b
; CHECK: fmov w0, s[[REDUCE]]
; CHECK: ret
  %res = call i8 @llvm.experimental.vector.reduce.and.v8i8(<8 x i8> %a)
  ret i8 %res
}

; No single instruction NEON ANDV support. Use SVE.
define i8 @andv_v16i8(<16 x i8> %a) #0 {
; CHECK-LABEL: andv_v16i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl16
; CHECK: andv b[[REDUCE:[0-9]+]], [[PG]], z0.b
; CHECK: fmov w0, s[[REDUCE]]
; CHECK: ret
  %res = call i8 @llvm.experimental.vector.reduce.and.v16i8(<16 x i8> %a)
  ret i8 %res
}

define i8 @andv_v32i8(<32 x i8>* %a) #0 {
; CHECK-LABEL: andv_v32i8:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].b, vl32
; VBITS_GE_256-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_256-NEXT: andv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_256-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_256-NEXT: ret
  %op = load <32 x i8>, <32 x i8>* %a
  %res = call i8 @llvm.experimental.vector.reduce.and.v32i8(<32 x i8> %op)
  ret i8 %res
}

define i8 @andv_v64i8(<64 x i8>* %a) #0 {
; CHECK-LABEL: andv_v64i8:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].b, vl64
; VBITS_GE_512-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: andv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_512-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].b, vl32
; VBITS_EQ_256-DAG: mov w[[A_HI:[0-9]+]], #32
; VBITS_EQ_256-DAG: ld1b { [[LO:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1b { [[HI:z[0-9]+]].b }, [[PG]]/z, [x0, x[[A_HI]]]
; VBITS_EQ_256-DAG: and [[AND:z[0-9]+]].d, [[LO]].d, [[HI]].d
; VBITS_EQ_256-DAG: andv b[[REDUCE:[0-9]+]], [[PG]], [[AND]].b
; VBITS_EQ_256-NEXT: fmov w0, s[[REDUCE]]
; VBITS_EQ_256-NEXT: ret

  %op = load <64 x i8>, <64 x i8>* %a
  %res = call i8 @llvm.experimental.vector.reduce.and.v64i8(<64 x i8> %op)
  ret i8 %res
}

define i8 @andv_v128i8(<128 x i8>* %a) #0 {
; CHECK-LABEL: andv_v128i8:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].b, vl128
; VBITS_GE_1024-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: andv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_1024-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <128 x i8>, <128 x i8>* %a
  %res = call i8 @llvm.experimental.vector.reduce.and.v128i8(<128 x i8> %op)
  ret i8 %res
}

define i8 @andv_v256i8(<256 x i8>* %a) #0 {
; CHECK-LABEL: andv_v256i8:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].b, vl256
; VBITS_GE_2048-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: andv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_2048-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <256 x i8>, <256 x i8>* %a
  %res = call i8 @llvm.experimental.vector.reduce.and.v256i8(<256 x i8> %op)
  ret i8 %res
}

; No single instruction NEON ANDV support. Use SVE.
define i16 @andv_v4i16(<4 x i16> %a) #0 {
; CHECK-LABEL: andv_v4i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl4
; CHECK: andv h[[REDUCE:[0-9]+]], [[PG]], z0.h
; CHECK: fmov w0, s[[REDUCE]]
; CHECK: ret
  %res = call i16 @llvm.experimental.vector.reduce.and.v4i16(<4 x i16> %a)
  ret i16 %res
}

; No single instruction NEON ANDV support. Use SVE.
define i16 @andv_v8i16(<8 x i16> %a) #0 {
; CHECK-LABEL: andv_v8i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl8
; CHECK: andv h[[REDUCE:[0-9]+]], [[PG]], z0.h
; CHECK: fmov w0, s[[REDUCE]]
; CHECK: ret
  %res = call i16 @llvm.experimental.vector.reduce.and.v8i16(<8 x i16> %a)
  ret i16 %res
}

define i16 @andv_v16i16(<16 x i16>* %a) #0 {
; CHECK-LABEL: andv_v16i16:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_GE_256-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_256-NEXT: andv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_256-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_256-NEXT: ret
  %op = load <16 x i16>, <16 x i16>* %a
  %res = call i16 @llvm.experimental.vector.reduce.and.v16i16(<16 x i16> %op)
  ret i16 %res
}

define i16 @andv_v32i16(<32 x i16>* %a) #0 {
; CHECK-LABEL: andv_v32i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: andv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_512-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1h { [[LO:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[HI:z[0-9]+]].h }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: and [[AND:z[0-9]+]].d, [[LO]].d, [[HI]].d
; VBITS_EQ_256-DAG: andv h[[REDUCE:[0-9]+]], [[PG]], [[AND]].h
; VBITS_EQ_256-NEXT: fmov w0, s[[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <32 x i16>, <32 x i16>* %a
  %res = call i16 @llvm.experimental.vector.reduce.and.v32i16(<32 x i16> %op)
  ret i16 %res
}

define i16 @andv_v64i16(<64 x i16>* %a) #0 {
; CHECK-LABEL: andv_v64i16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: andv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_1024-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <64 x i16>, <64 x i16>* %a
  %res = call i16 @llvm.experimental.vector.reduce.and.v64i16(<64 x i16> %op)
  ret i16 %res
}

define i16 @andv_v128i16(<128 x i16>* %a) #0 {
; CHECK-LABEL: andv_v128i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: andv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_2048-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <128 x i16>, <128 x i16>* %a
  %res = call i16 @llvm.experimental.vector.reduce.and.v128i16(<128 x i16> %op)
  ret i16 %res
}

; No single instruction NEON ANDV support. Use SVE.
define i32 @andv_v2i32(<2 x i32> %a) #0 {
; CHECK-LABEL: andv_v2i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl2
; CHECK: andv [[REDUCE:s[0-9]+]], [[PG]], z0.s
; CHECK: fmov w0, [[REDUCE]]
; CHECK: ret
  %res = call i32 @llvm.experimental.vector.reduce.and.v2i32(<2 x i32> %a)
  ret i32 %res
}

; No single instruction NEON ANDV support. Use SVE.
define i32 @andv_v4i32(<4 x i32> %a) #0 {
; CHECK-LABEL: andv_v4i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl4
; CHECK: andv [[REDUCE:s[0-9]+]], [[PG]], z0.s
; CHECK: fmov w0, [[REDUCE]]
; CHECK: ret
  %res = call i32 @llvm.experimental.vector.reduce.and.v4i32(<4 x i32> %a)
  ret i32 %res
}

define i32 @andv_v8i32(<8 x i32>* %a) #0 {
; CHECK-LABEL: andv_v8i32:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_GE_256-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_256-NEXT: andv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_256-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_256-NEXT: ret
  %op = load <8 x i32>, <8 x i32>* %a
  %res = call i32 @llvm.experimental.vector.reduce.and.v8i32(<8 x i32> %op)
  ret i32 %res
}

define i32 @andv_v16i32(<16 x i32>* %a) #0 {
; CHECK-LABEL: andv_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: andv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_512-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1w { [[LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[HI:z[0-9]+]].s }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: and [[AND:z[0-9]+]].d, [[LO]].d, [[HI]].d
; VBITS_EQ_256-DAG: andv [[REDUCE:s[0-9]+]], [[PG]], [[AND]].s
; VBITS_EQ_256-NEXT: fmov w0, [[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <16 x i32>, <16 x i32>* %a
  %res = call i32 @llvm.experimental.vector.reduce.and.v16i32(<16 x i32> %op)
  ret i32 %res
}

define i32 @andv_v32i32(<32 x i32>* %a) #0 {
; CHECK-LABEL: andv_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: andv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_1024-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <32 x i32>, <32 x i32>* %a
  %res = call i32 @llvm.experimental.vector.reduce.and.v32i32(<32 x i32> %op)
  ret i32 %res
}

define i32 @andv_v64i32(<64 x i32>* %a) #0 {
; CHECK-LABEL: andv_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: andv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_2048-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <64 x i32>, <64 x i32>* %a
  %res = call i32 @llvm.experimental.vector.reduce.and.v64i32(<64 x i32> %op)
  ret i32 %res
}

; Nothing to do for single element vectors.
define i64 @andv_v1i64(<1 x i64> %a) #0 {
; CHECK-LABEL: andv_v1i64:
; CHECK: fmov x0, d0
; CHECK: ret
  %res = call i64 @llvm.experimental.vector.reduce.and.v1i64(<1 x i64> %a)
  ret i64 %res
}

; Use SVE for 128-bit vectors
define i64 @andv_v2i64(<2 x i64> %a) #0 {
; CHECK-LABEL: andv_v2i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl2
; CHECK: andv [[REDUCE:d[0-9]+]], [[PG]], z0.d
; CHECK: fmov x0, [[REDUCE]]
; CHECK: ret
  %res = call i64 @llvm.experimental.vector.reduce.and.v2i64(<2 x i64> %a)
  ret i64 %res
}

define i64 @andv_v4i64(<4 x i64>* %a) #0 {
; CHECK-LABEL: andv_v4i64:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_GE_256-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_256-NEXT: andv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_256-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_256-NEXT: ret
  %op = load <4 x i64>, <4 x i64>* %a
  %res = call i64 @llvm.experimental.vector.reduce.and.v4i64(<4 x i64> %op)
  ret i64 %res
}

define i64 @andv_v8i64(<8 x i64>* %a) #0 {
; CHECK-LABEL: andv_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: andv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_512-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1d { [[LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[HI:z[0-9]+]].d }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: and [[AND:z[0-9]+]].d, [[LO]].d, [[HI]].d
; VBITS_EQ_256-DAG: andv [[REDUCE:d[0-9]+]], [[PG]], [[AND]].d
; VBITS_EQ_256-NEXT: fmov x0, [[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <8 x i64>, <8 x i64>* %a
  %res = call i64 @llvm.experimental.vector.reduce.and.v8i64(<8 x i64> %op)
  ret i64 %res
}

define i64 @andv_v16i64(<16 x i64>* %a) #0 {
; CHECK-LABEL: andv_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: andv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_1024-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <16 x i64>, <16 x i64>* %a
  %res = call i64 @llvm.experimental.vector.reduce.and.v16i64(<16 x i64> %op)
  ret i64 %res
}

define i64 @andv_v32i64(<32 x i64>* %a) #0 {
; CHECK-LABEL: andv_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: andv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_2048-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <32 x i64>, <32 x i64>* %a
  %res = call i64 @llvm.experimental.vector.reduce.and.v32i64(<32 x i64> %op)
  ret i64 %res
}

;
; EORV
;

; No single instruction NEON EORV support. Use SVE.
define i8 @eorv_v8i8(<8 x i8> %a) #0 {
; CHECK-LABEL: eorv_v8i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl8
; CHECK: eorv b[[REDUCE:[0-9]+]], [[PG]], z0.b
; CHECK: fmov w0, s[[REDUCE]]
; CHECK: ret
  %res = call i8 @llvm.experimental.vector.reduce.xor.v8i8(<8 x i8> %a)
  ret i8 %res
}

; No single instruction NEON EORV support. Use SVE.
define i8 @eorv_v16i8(<16 x i8> %a) #0 {
; CHECK-LABEL: eorv_v16i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl16
; CHECK: eorv b[[REDUCE:[0-9]+]], [[PG]], z0.b
; CHECK: fmov w0, s[[REDUCE]]
; CHECK: ret
  %res = call i8 @llvm.experimental.vector.reduce.xor.v16i8(<16 x i8> %a)
  ret i8 %res
}

define i8 @eorv_v32i8(<32 x i8>* %a) #0 {
; CHECK-LABEL: eorv_v32i8:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].b, vl32
; VBITS_GE_256-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_256-NEXT: eorv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_256-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_256-NEXT: ret
  %op = load <32 x i8>, <32 x i8>* %a
  %res = call i8 @llvm.experimental.vector.reduce.xor.v32i8(<32 x i8> %op)
  ret i8 %res
}

define i8 @eorv_v64i8(<64 x i8>* %a) #0 {
; CHECK-LABEL: eorv_v64i8:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].b, vl64
; VBITS_GE_512-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: eorv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_512-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].b, vl32
; VBITS_EQ_256-DAG: mov w[[A_HI:[0-9]+]], #32
; VBITS_EQ_256-DAG: ld1b { [[LO:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1b { [[HI:z[0-9]+]].b }, [[PG]]/z, [x0, x[[A_HI]]]
; VBITS_EQ_256-DAG: eor [[EOR:z[0-9]+]].d, [[LO]].d, [[HI]].d
; VBITS_EQ_256-DAG: eorv b[[REDUCE:[0-9]+]], [[PG]], [[EOR]].b
; VBITS_EQ_256-NEXT: fmov w0, s[[REDUCE]]
; VBITS_EQ_256-NEXT: ret

  %op = load <64 x i8>, <64 x i8>* %a
  %res = call i8 @llvm.experimental.vector.reduce.xor.v64i8(<64 x i8> %op)
  ret i8 %res
}

define i8 @eorv_v128i8(<128 x i8>* %a) #0 {
; CHECK-LABEL: eorv_v128i8:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].b, vl128
; VBITS_GE_1024-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: eorv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_1024-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <128 x i8>, <128 x i8>* %a
  %res = call i8 @llvm.experimental.vector.reduce.xor.v128i8(<128 x i8> %op)
  ret i8 %res
}

define i8 @eorv_v256i8(<256 x i8>* %a) #0 {
; CHECK-LABEL: eorv_v256i8:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].b, vl256
; VBITS_GE_2048-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: eorv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_2048-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <256 x i8>, <256 x i8>* %a
  %res = call i8 @llvm.experimental.vector.reduce.xor.v256i8(<256 x i8> %op)
  ret i8 %res
}

; No single instruction NEON EORV support. Use SVE.
define i16 @eorv_v4i16(<4 x i16> %a) #0 {
; CHECK-LABEL: eorv_v4i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl4
; CHECK: eorv h[[REDUCE:[0-9]+]], [[PG]], z0.h
; CHECK: fmov w0, s[[REDUCE]]
; CHECK: ret
  %res = call i16 @llvm.experimental.vector.reduce.xor.v4i16(<4 x i16> %a)
  ret i16 %res
}

; No single instruction NEON EORV support. Use SVE.
define i16 @eorv_v8i16(<8 x i16> %a) #0 {
; CHECK-LABEL: eorv_v8i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl8
; CHECK: eorv h[[REDUCE:[0-9]+]], [[PG]], z0.h
; CHECK: fmov w0, s[[REDUCE]]
; CHECK: ret
  %res = call i16 @llvm.experimental.vector.reduce.xor.v8i16(<8 x i16> %a)
  ret i16 %res
}

define i16 @eorv_v16i16(<16 x i16>* %a) #0 {
; CHECK-LABEL: eorv_v16i16:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_GE_256-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_256-NEXT: eorv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_256-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_256-NEXT: ret
  %op = load <16 x i16>, <16 x i16>* %a
  %res = call i16 @llvm.experimental.vector.reduce.xor.v16i16(<16 x i16> %op)
  ret i16 %res
}

define i16 @eorv_v32i16(<32 x i16>* %a) #0 {
; CHECK-LABEL: eorv_v32i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: eorv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_512-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1h { [[LO:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[HI:z[0-9]+]].h }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: eor [[EOR:z[0-9]+]].d, [[LO]].d, [[HI]].d
; VBITS_EQ_256-DAG: eorv h[[REDUCE:[0-9]+]], [[PG]], [[EOR]].h
; VBITS_EQ_256-NEXT: fmov w0, s[[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <32 x i16>, <32 x i16>* %a
  %res = call i16 @llvm.experimental.vector.reduce.xor.v32i16(<32 x i16> %op)
  ret i16 %res
}

define i16 @eorv_v64i16(<64 x i16>* %a) #0 {
; CHECK-LABEL: eorv_v64i16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: eorv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_1024-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <64 x i16>, <64 x i16>* %a
  %res = call i16 @llvm.experimental.vector.reduce.xor.v64i16(<64 x i16> %op)
  ret i16 %res
}

define i16 @eorv_v128i16(<128 x i16>* %a) #0 {
; CHECK-LABEL: eorv_v128i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: eorv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_2048-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <128 x i16>, <128 x i16>* %a
  %res = call i16 @llvm.experimental.vector.reduce.xor.v128i16(<128 x i16> %op)
  ret i16 %res
}

; No single instruction NEON EORV support. Use SVE.
define i32 @eorv_v2i32(<2 x i32> %a) #0 {
; CHECK-LABEL: eorv_v2i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl2
; CHECK: eorv [[REDUCE:s[0-9]+]], [[PG]], z0.s
; CHECK: fmov w0, [[REDUCE]]
; CHECK: ret
  %res = call i32 @llvm.experimental.vector.reduce.xor.v2i32(<2 x i32> %a)
  ret i32 %res
}

; No single instruction NEON EORV support. Use SVE.
define i32 @eorv_v4i32(<4 x i32> %a) #0 {
; CHECK-LABEL: eorv_v4i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl4
; CHECK: eorv [[REDUCE:s[0-9]+]], [[PG]], z0.s
; CHECK: fmov w0, [[REDUCE]]
; CHECK: ret
  %res = call i32 @llvm.experimental.vector.reduce.xor.v4i32(<4 x i32> %a)
  ret i32 %res
}

define i32 @eorv_v8i32(<8 x i32>* %a) #0 {
; CHECK-LABEL: eorv_v8i32:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_GE_256-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_256-NEXT: eorv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_256-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_256-NEXT: ret
  %op = load <8 x i32>, <8 x i32>* %a
  %res = call i32 @llvm.experimental.vector.reduce.xor.v8i32(<8 x i32> %op)
  ret i32 %res
}

define i32 @eorv_v16i32(<16 x i32>* %a) #0 {
; CHECK-LABEL: eorv_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: eorv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_512-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1w { [[LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[HI:z[0-9]+]].s }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: eor [[EOR:z[0-9]+]].d, [[LO]].d, [[HI]].d
; VBITS_EQ_256-DAG: eorv [[REDUCE:s[0-9]+]], [[PG]], [[EOR]].s
; VBITS_EQ_256-NEXT: fmov w0, [[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <16 x i32>, <16 x i32>* %a
  %res = call i32 @llvm.experimental.vector.reduce.xor.v16i32(<16 x i32> %op)
  ret i32 %res
}

define i32 @eorv_v32i32(<32 x i32>* %a) #0 {
; CHECK-LABEL: eorv_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: eorv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_1024-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <32 x i32>, <32 x i32>* %a
  %res = call i32 @llvm.experimental.vector.reduce.xor.v32i32(<32 x i32> %op)
  ret i32 %res
}

define i32 @eorv_v64i32(<64 x i32>* %a) #0 {
; CHECK-LABEL: eorv_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: eorv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_2048-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <64 x i32>, <64 x i32>* %a
  %res = call i32 @llvm.experimental.vector.reduce.xor.v64i32(<64 x i32> %op)
  ret i32 %res
}

; Nothing to do for single element vectors.
define i64 @eorv_v1i64(<1 x i64> %a) #0 {
; CHECK-LABEL: eorv_v1i64:
; CHECK: fmov x0, d0
; CHECK: ret
  %res = call i64 @llvm.experimental.vector.reduce.xor.v1i64(<1 x i64> %a)
  ret i64 %res
}

; Use SVE for 128-bit vectors
define i64 @eorv_v2i64(<2 x i64> %a) #0 {
; CHECK-LABEL: eorv_v2i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl2
; CHECK: eorv [[REDUCE:d[0-9]+]], [[PG]], z0.d
; CHECK: fmov x0, [[REDUCE]]
; CHECK: ret
  %res = call i64 @llvm.experimental.vector.reduce.xor.v2i64(<2 x i64> %a)
  ret i64 %res
}

define i64 @eorv_v4i64(<4 x i64>* %a) #0 {
; CHECK-LABEL: eorv_v4i64:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_GE_256-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_256-NEXT: eorv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_256-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_256-NEXT: ret
  %op = load <4 x i64>, <4 x i64>* %a
  %res = call i64 @llvm.experimental.vector.reduce.xor.v4i64(<4 x i64> %op)
  ret i64 %res
}

define i64 @eorv_v8i64(<8 x i64>* %a) #0 {
; CHECK-LABEL: eorv_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: eorv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_512-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1d { [[LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[HI:z[0-9]+]].d }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: eor [[EOR:z[0-9]+]].d, [[LO]].d, [[HI]].d
; VBITS_EQ_256-DAG: eorv [[REDUCE:d[0-9]+]], [[PG]], [[EOR]].d
; VBITS_EQ_256-NEXT: fmov x0, [[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <8 x i64>, <8 x i64>* %a
  %res = call i64 @llvm.experimental.vector.reduce.xor.v8i64(<8 x i64> %op)
  ret i64 %res
}

define i64 @eorv_v16i64(<16 x i64>* %a) #0 {
; CHECK-LABEL: eorv_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: eorv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_1024-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <16 x i64>, <16 x i64>* %a
  %res = call i64 @llvm.experimental.vector.reduce.xor.v16i64(<16 x i64> %op)
  ret i64 %res
}

define i64 @eorv_v32i64(<32 x i64>* %a) #0 {
; CHECK-LABEL: eorv_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: eorv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_2048-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <32 x i64>, <32 x i64>* %a
  %res = call i64 @llvm.experimental.vector.reduce.xor.v32i64(<32 x i64> %op)
  ret i64 %res
}

;
; ORV
;

; No single instruction NEON ORV support. Use SVE.
define i8 @orv_v8i8(<8 x i8> %a) #0 {
; CHECK-LABEL: orv_v8i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl8
; CHECK: orv b[[REDUCE:[0-9]+]], [[PG]], z0.b
; CHECK: fmov w0, s[[REDUCE]]
; CHECK: ret
  %res = call i8 @llvm.experimental.vector.reduce.or.v8i8(<8 x i8> %a)
  ret i8 %res
}

; No single instruction NEON ORV support. Use SVE.
define i8 @orv_v16i8(<16 x i8> %a) #0 {
; CHECK-LABEL: orv_v16i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl16
; CHECK: orv b[[REDUCE:[0-9]+]], [[PG]], z0.b
; CHECK: fmov w0, s[[REDUCE]]
; CHECK: ret
  %res = call i8 @llvm.experimental.vector.reduce.or.v16i8(<16 x i8> %a)
  ret i8 %res
}

define i8 @orv_v32i8(<32 x i8>* %a) #0 {
; CHECK-LABEL: orv_v32i8:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].b, vl32
; VBITS_GE_256-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_256-NEXT: orv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_256-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_256-NEXT: ret
  %op = load <32 x i8>, <32 x i8>* %a
  %res = call i8 @llvm.experimental.vector.reduce.or.v32i8(<32 x i8> %op)
  ret i8 %res
}

define i8 @orv_v64i8(<64 x i8>* %a) #0 {
; CHECK-LABEL: orv_v64i8:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].b, vl64
; VBITS_GE_512-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: orv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_512-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].b, vl32
; VBITS_EQ_256-DAG: mov w[[A_HI:[0-9]+]], #32
; VBITS_EQ_256-DAG: ld1b { [[LO:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1b { [[HI:z[0-9]+]].b }, [[PG]]/z, [x0, x[[A_HI]]]
; VBITS_EQ_256-DAG: orr [[OR:z[0-9]+]].d, [[LO]].d, [[HI]].d
; VBITS_EQ_256-DAG: orv b[[REDUCE:[0-9]+]], [[PG]], [[OR]].b
; VBITS_EQ_256-NEXT: fmov w0, s[[REDUCE]]
; VBITS_EQ_256-NEXT: ret

  %op = load <64 x i8>, <64 x i8>* %a
  %res = call i8 @llvm.experimental.vector.reduce.or.v64i8(<64 x i8> %op)
  ret i8 %res
}

define i8 @orv_v128i8(<128 x i8>* %a) #0 {
; CHECK-LABEL: orv_v128i8:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].b, vl128
; VBITS_GE_1024-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: orv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_1024-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <128 x i8>, <128 x i8>* %a
  %res = call i8 @llvm.experimental.vector.reduce.or.v128i8(<128 x i8> %op)
  ret i8 %res
}

define i8 @orv_v256i8(<256 x i8>* %a) #0 {
; CHECK-LABEL: orv_v256i8:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].b, vl256
; VBITS_GE_2048-NEXT: ld1b { [[OP:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: orv b[[REDUCE:[0-9]+]], [[PG]], [[OP]].b
; VBITS_GE_2048-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <256 x i8>, <256 x i8>* %a
  %res = call i8 @llvm.experimental.vector.reduce.or.v256i8(<256 x i8> %op)
  ret i8 %res
}

; No single instruction NEON ORV support. Use SVE.
define i16 @orv_v4i16(<4 x i16> %a) #0 {
; CHECK-LABEL: orv_v4i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl4
; CHECK: orv h[[REDUCE:[0-9]+]], [[PG]], z0.h
; CHECK: fmov w0, s[[REDUCE]]
; CHECK: ret
  %res = call i16 @llvm.experimental.vector.reduce.or.v4i16(<4 x i16> %a)
  ret i16 %res
}

; No single instruction NEON ORV support. Use SVE.
define i16 @orv_v8i16(<8 x i16> %a) #0 {
; CHECK-LABEL: orv_v8i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl8
; CHECK: orv h[[REDUCE:[0-9]+]], [[PG]], z0.h
; CHECK: fmov w0, s[[REDUCE]]
; CHECK: ret
  %res = call i16 @llvm.experimental.vector.reduce.or.v8i16(<8 x i16> %a)
  ret i16 %res
}

define i16 @orv_v16i16(<16 x i16>* %a) #0 {
; CHECK-LABEL: orv_v16i16:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_GE_256-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_256-NEXT: orv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_256-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_256-NEXT: ret
  %op = load <16 x i16>, <16 x i16>* %a
  %res = call i16 @llvm.experimental.vector.reduce.or.v16i16(<16 x i16> %op)
  ret i16 %res
}

define i16 @orv_v32i16(<32 x i16>* %a) #0 {
; CHECK-LABEL: orv_v32i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: orv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_512-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1h { [[LO:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[HI:z[0-9]+]].h }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: orr [[OR:z[0-9]+]].d, [[LO]].d, [[HI]].d
; VBITS_EQ_256-DAG: orv h[[REDUCE:[0-9]+]], [[PG]], [[OR]].h
; VBITS_EQ_256-NEXT: fmov w0, s[[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <32 x i16>, <32 x i16>* %a
  %res = call i16 @llvm.experimental.vector.reduce.or.v32i16(<32 x i16> %op)
  ret i16 %res
}

define i16 @orv_v64i16(<64 x i16>* %a) #0 {
; CHECK-LABEL: orv_v64i16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: orv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_1024-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <64 x i16>, <64 x i16>* %a
  %res = call i16 @llvm.experimental.vector.reduce.or.v64i16(<64 x i16> %op)
  ret i16 %res
}

define i16 @orv_v128i16(<128 x i16>* %a) #0 {
; CHECK-LABEL: orv_v128i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: orv h[[REDUCE:[0-9]+]], [[PG]], [[OP]].h
; VBITS_GE_2048-NEXT: fmov w0, s[[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <128 x i16>, <128 x i16>* %a
  %res = call i16 @llvm.experimental.vector.reduce.or.v128i16(<128 x i16> %op)
  ret i16 %res
}

; No single instruction NEON ORV support. Use SVE.
define i32 @orv_v2i32(<2 x i32> %a) #0 {
; CHECK-LABEL: orv_v2i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl2
; CHECK: orv [[REDUCE:s[0-9]+]], [[PG]], z0.s
; CHECK: fmov w0, [[REDUCE]]
; CHECK: ret
  %res = call i32 @llvm.experimental.vector.reduce.or.v2i32(<2 x i32> %a)
  ret i32 %res
}

; No single instruction NEON ORV support. Use SVE.
define i32 @orv_v4i32(<4 x i32> %a) #0 {
; CHECK-LABEL: orv_v4i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl4
; CHECK: orv [[REDUCE:s[0-9]+]], [[PG]], z0.s
; CHECK: fmov w0, [[REDUCE]]
; CHECK: ret
  %res = call i32 @llvm.experimental.vector.reduce.or.v4i32(<4 x i32> %a)
  ret i32 %res
}

define i32 @orv_v8i32(<8 x i32>* %a) #0 {
; CHECK-LABEL: orv_v8i32:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_GE_256-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_256-NEXT: orv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_256-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_256-NEXT: ret
  %op = load <8 x i32>, <8 x i32>* %a
  %res = call i32 @llvm.experimental.vector.reduce.or.v8i32(<8 x i32> %op)
  ret i32 %res
}

define i32 @orv_v16i32(<16 x i32>* %a) #0 {
; CHECK-LABEL: orv_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: orv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_512-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1w { [[LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[HI:z[0-9]+]].s }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: orr [[OR:z[0-9]+]].d, [[LO]].d, [[HI]].d
; VBITS_EQ_256-DAG: orv [[REDUCE:s[0-9]+]], [[PG]], [[OR]].s
; VBITS_EQ_256-NEXT: fmov w0, [[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <16 x i32>, <16 x i32>* %a
  %res = call i32 @llvm.experimental.vector.reduce.or.v16i32(<16 x i32> %op)
  ret i32 %res
}

define i32 @orv_v32i32(<32 x i32>* %a) #0 {
; CHECK-LABEL: orv_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: orv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_1024-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <32 x i32>, <32 x i32>* %a
  %res = call i32 @llvm.experimental.vector.reduce.or.v32i32(<32 x i32> %op)
  ret i32 %res
}

define i32 @orv_v64i32(<64 x i32>* %a) #0 {
; CHECK-LABEL: orv_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: orv [[REDUCE:s[0-9]+]], [[PG]], [[OP]].s
; VBITS_GE_2048-NEXT: fmov w0, [[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <64 x i32>, <64 x i32>* %a
  %res = call i32 @llvm.experimental.vector.reduce.or.v64i32(<64 x i32> %op)
  ret i32 %res
}

; Nothing to do for single element vectors.
define i64 @orv_v1i64(<1 x i64> %a) #0 {
; CHECK-LABEL: orv_v1i64:
; CHECK: fmov x0, d0
; CHECK: ret
  %res = call i64 @llvm.experimental.vector.reduce.or.v1i64(<1 x i64> %a)
  ret i64 %res
}

; Use SVE for 128-bit vectors
define i64 @orv_v2i64(<2 x i64> %a) #0 {
; CHECK-LABEL: orv_v2i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl2
; CHECK: orv [[REDUCE:d[0-9]+]], [[PG]], z0.d
; CHECK: fmov x0, [[REDUCE]]
; CHECK: ret
  %res = call i64 @llvm.experimental.vector.reduce.or.v2i64(<2 x i64> %a)
  ret i64 %res
}

define i64 @orv_v4i64(<4 x i64>* %a) #0 {
; CHECK-LABEL: orv_v4i64:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_GE_256-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_256-NEXT: orv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_256-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_256-NEXT: ret
  %op = load <4 x i64>, <4 x i64>* %a
  %res = call i64 @llvm.experimental.vector.reduce.or.v4i64(<4 x i64> %op)
  ret i64 %res
}

define i64 @orv_v8i64(<8 x i64>* %a) #0 {
; CHECK-LABEL: orv_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: orv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_512-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1d { [[LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[HI:z[0-9]+]].d }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: orr [[OR:z[0-9]+]].d, [[LO]].d, [[HI]].d
; VBITS_EQ_256-DAG: orv [[REDUCE:d[0-9]+]], [[PG]], [[OR]].d
; VBITS_EQ_256-NEXT: fmov x0, [[REDUCE]]
; VBITS_EQ_256-NEXT: ret
  %op = load <8 x i64>, <8 x i64>* %a
  %res = call i64 @llvm.experimental.vector.reduce.or.v8i64(<8 x i64> %op)
  ret i64 %res
}

define i64 @orv_v16i64(<16 x i64>* %a) #0 {
; CHECK-LABEL: orv_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: orv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_1024-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_1024-NEXT: ret
  %op = load <16 x i64>, <16 x i64>* %a
  %res = call i64 @llvm.experimental.vector.reduce.or.v16i64(<16 x i64> %op)
  ret i64 %res
}

define i64 @orv_v32i64(<32 x i64>* %a) #0 {
; CHECK-LABEL: orv_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: orv [[REDUCE:d[0-9]+]], [[PG]], [[OP]].d
; VBITS_GE_2048-NEXT: fmov x0, [[REDUCE]]
; VBITS_GE_2048-NEXT: ret
  %op = load <32 x i64>, <32 x i64>* %a
  %res = call i64 @llvm.experimental.vector.reduce.or.v32i64(<32 x i64> %op)
  ret i64 %res
}

attributes #0 = { "target-features"="+sve" }

declare i8 @llvm.experimental.vector.reduce.and.v8i8(<8 x i8>)
declare i8 @llvm.experimental.vector.reduce.and.v16i8(<16 x i8>)
declare i8 @llvm.experimental.vector.reduce.and.v32i8(<32 x i8>)
declare i8 @llvm.experimental.vector.reduce.and.v64i8(<64 x i8>)
declare i8 @llvm.experimental.vector.reduce.and.v128i8(<128 x i8>)
declare i8 @llvm.experimental.vector.reduce.and.v256i8(<256 x i8>)

declare i16 @llvm.experimental.vector.reduce.and.v4i16(<4 x i16>)
declare i16 @llvm.experimental.vector.reduce.and.v8i16(<8 x i16>)
declare i16 @llvm.experimental.vector.reduce.and.v16i16(<16 x i16>)
declare i16 @llvm.experimental.vector.reduce.and.v32i16(<32 x i16>)
declare i16 @llvm.experimental.vector.reduce.and.v64i16(<64 x i16>)
declare i16 @llvm.experimental.vector.reduce.and.v128i16(<128 x i16>)

declare i32 @llvm.experimental.vector.reduce.and.v2i32(<2 x i32>)
declare i32 @llvm.experimental.vector.reduce.and.v4i32(<4 x i32>)
declare i32 @llvm.experimental.vector.reduce.and.v8i32(<8 x i32>)
declare i32 @llvm.experimental.vector.reduce.and.v16i32(<16 x i32>)
declare i32 @llvm.experimental.vector.reduce.and.v32i32(<32 x i32>)
declare i32 @llvm.experimental.vector.reduce.and.v64i32(<64 x i32>)

declare i64 @llvm.experimental.vector.reduce.and.v1i64(<1 x i64>)
declare i64 @llvm.experimental.vector.reduce.and.v2i64(<2 x i64>)
declare i64 @llvm.experimental.vector.reduce.and.v4i64(<4 x i64>)
declare i64 @llvm.experimental.vector.reduce.and.v8i64(<8 x i64>)
declare i64 @llvm.experimental.vector.reduce.and.v16i64(<16 x i64>)
declare i64 @llvm.experimental.vector.reduce.and.v32i64(<32 x i64>)

declare i8 @llvm.experimental.vector.reduce.or.v8i8(<8 x i8>)
declare i8 @llvm.experimental.vector.reduce.or.v16i8(<16 x i8>)
declare i8 @llvm.experimental.vector.reduce.or.v32i8(<32 x i8>)
declare i8 @llvm.experimental.vector.reduce.or.v64i8(<64 x i8>)
declare i8 @llvm.experimental.vector.reduce.or.v128i8(<128 x i8>)
declare i8 @llvm.experimental.vector.reduce.or.v256i8(<256 x i8>)

declare i16 @llvm.experimental.vector.reduce.or.v4i16(<4 x i16>)
declare i16 @llvm.experimental.vector.reduce.or.v8i16(<8 x i16>)
declare i16 @llvm.experimental.vector.reduce.or.v16i16(<16 x i16>)
declare i16 @llvm.experimental.vector.reduce.or.v32i16(<32 x i16>)
declare i16 @llvm.experimental.vector.reduce.or.v64i16(<64 x i16>)
declare i16 @llvm.experimental.vector.reduce.or.v128i16(<128 x i16>)

declare i32 @llvm.experimental.vector.reduce.or.v2i32(<2 x i32>)
declare i32 @llvm.experimental.vector.reduce.or.v4i32(<4 x i32>)
declare i32 @llvm.experimental.vector.reduce.or.v8i32(<8 x i32>)
declare i32 @llvm.experimental.vector.reduce.or.v16i32(<16 x i32>)
declare i32 @llvm.experimental.vector.reduce.or.v32i32(<32 x i32>)
declare i32 @llvm.experimental.vector.reduce.or.v64i32(<64 x i32>)

declare i64 @llvm.experimental.vector.reduce.or.v1i64(<1 x i64>)
declare i64 @llvm.experimental.vector.reduce.or.v2i64(<2 x i64>)
declare i64 @llvm.experimental.vector.reduce.or.v4i64(<4 x i64>)
declare i64 @llvm.experimental.vector.reduce.or.v8i64(<8 x i64>)
declare i64 @llvm.experimental.vector.reduce.or.v16i64(<16 x i64>)
declare i64 @llvm.experimental.vector.reduce.or.v32i64(<32 x i64>)

declare i8 @llvm.experimental.vector.reduce.xor.v8i8(<8 x i8>)
declare i8 @llvm.experimental.vector.reduce.xor.v16i8(<16 x i8>)
declare i8 @llvm.experimental.vector.reduce.xor.v32i8(<32 x i8>)
declare i8 @llvm.experimental.vector.reduce.xor.v64i8(<64 x i8>)
declare i8 @llvm.experimental.vector.reduce.xor.v128i8(<128 x i8>)
declare i8 @llvm.experimental.vector.reduce.xor.v256i8(<256 x i8>)

declare i16 @llvm.experimental.vector.reduce.xor.v4i16(<4 x i16>)
declare i16 @llvm.experimental.vector.reduce.xor.v8i16(<8 x i16>)
declare i16 @llvm.experimental.vector.reduce.xor.v16i16(<16 x i16>)
declare i16 @llvm.experimental.vector.reduce.xor.v32i16(<32 x i16>)
declare i16 @llvm.experimental.vector.reduce.xor.v64i16(<64 x i16>)
declare i16 @llvm.experimental.vector.reduce.xor.v128i16(<128 x i16>)

declare i32 @llvm.experimental.vector.reduce.xor.v2i32(<2 x i32>)
declare i32 @llvm.experimental.vector.reduce.xor.v4i32(<4 x i32>)
declare i32 @llvm.experimental.vector.reduce.xor.v8i32(<8 x i32>)
declare i32 @llvm.experimental.vector.reduce.xor.v16i32(<16 x i32>)
declare i32 @llvm.experimental.vector.reduce.xor.v32i32(<32 x i32>)
declare i32 @llvm.experimental.vector.reduce.xor.v64i32(<64 x i32>)

declare i64 @llvm.experimental.vector.reduce.xor.v1i64(<1 x i64>)
declare i64 @llvm.experimental.vector.reduce.xor.v2i64(<2 x i64>)
declare i64 @llvm.experimental.vector.reduce.xor.v4i64(<4 x i64>)
declare i64 @llvm.experimental.vector.reduce.xor.v8i64(<8 x i64>)
declare i64 @llvm.experimental.vector.reduce.xor.v16i64(<16 x i64>)
declare i64 @llvm.experimental.vector.reduce.xor.v32i64(<32 x i64>)
