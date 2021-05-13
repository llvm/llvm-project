; RUN: llc -aarch64-sve-vector-bits-min=128  -asm-verbose=0 < %s | FileCheck %s -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=384  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=512  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=640  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=768  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=896  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1024 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1152 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1280 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1408 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1536 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1664 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1792 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1920 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=2048 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_GE_2048

target triple = "aarch64-unknown-linux-gnu"

; Don't use SVE when its registers are no bigger than NEON.
; NO_SVE-NOT: ptrue

; Don't use SVE for 64-bit vectors.
define void @bitcast_v4i16(<4 x i16> *%a, <4 x half>* %b) #0 {
; CHECK-LABEL: bitcast_v4i16:
; CHECK: ldr d0, [x0]
; CHECK-NEXT: str d0, [x1]
; CHECK-NEXT: ret
  %load = load volatile <4 x i16>, <4 x i16>* %a
  %cast = bitcast <4 x i16> %load to <4 x half>
  store volatile <4 x half> %cast, <4 x half>* %b
  ret void
}

; Don't use SVE for 128-bit vectors.
define void @bitcast_v8i16(<8 x i16> *%a, <8 x half>* %b) #0 {
; CHECK-LABEL: bitcast_v8i16:
; CHECK: ldr q0, [x0]
; CHECK-NEXT: str q0, [x1]
; CHECK-NEXT: ret
  %load = load volatile <8 x i16>, <8 x i16>* %a
  %cast = bitcast <8 x i16> %load to <8 x half>
  store volatile <8 x half> %cast, <8 x half>* %b
  ret void
}

define void @bitcast_v16i16(<16 x i16> *%a, <16 x half>* %b) #0 {
; CHECK-LABEL: bitcast_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-NEXT: st1h { [[OP]].h }, [[PG]], [x1]
; CHECK-NEXT: ret
  %load = load volatile <16 x i16>, <16 x i16>* %a
  %cast = bitcast <16 x i16> %load to <16 x half>
  store volatile <16 x half> %cast, <16 x half>* %b
  ret void
}

define void @bitcast_v32i16(<32 x i16> *%a, <32 x half>* %b) #0 {
; CHECK-LABEL: bitcast_v32i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: st1h { [[OP]].h }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret
  %load = load volatile <32 x i16>, <32 x i16>* %a
  %cast = bitcast <32 x i16> %load to <32 x half>
  store volatile <32 x half> %cast, <32 x half>* %b
  ret void
}

define void @bitcast_v64i16(<64 x i16> *%a, <64 x half>* %b) #0 {
; CHECK-LABEL: bitcast_v64i16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: st1h { [[OP]].h }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %load = load volatile <64 x i16>, <64 x i16>* %a
  %cast = bitcast <64 x i16> %load to <64 x half>
  store volatile <64 x half> %cast, <64 x half>* %b
  ret void
}

define void @bitcast_v128i16(<128 x i16> *%a, <128 x half>* %b) #0 {
; CHECK-LABEL: bitcast_v128i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: st1h { [[OP]].h }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %load = load volatile <128 x i16>, <128 x i16>* %a
  %cast = bitcast <128 x i16> %load to <128 x half>
  store volatile <128 x half> %cast, <128 x half>* %b
  ret void
}

; Don't use SVE for 64-bit vectors.
define void @bitcast_v2i32(<2 x i32> *%a, <2 x float>* %b) #0 {
; CHECK-LABEL: bitcast_v2i32:
; CHECK: ldr d0, [x0]
; CHECK-NEXT: str d0, [x1]
; CHECK-NEXT: ret
  %load = load volatile <2 x i32>, <2 x i32>* %a
  %cast = bitcast <2 x i32> %load to <2 x float>
  store volatile <2 x float> %cast, <2 x float>* %b
  ret void
}

; Don't use SVE for 128-bit vectors.
define void @bitcast_v4i32(<4 x i32> *%a, <4 x float>* %b) #0 {
; CHECK-LABEL: bitcast_v4i32:
; CHECK: ldr q0, [x0]
; CHECK-NEXT: str q0, [x1]
; CHECK-NEXT: ret
  %load = load volatile <4 x i32>, <4 x i32>* %a
  %cast = bitcast <4 x i32> %load to <4 x float>
  store volatile <4 x float> %cast, <4 x float>* %b
  ret void
}

define void @bitcast_v8i32(<8 x i32> *%a, <8 x float>* %b) #0 {
; CHECK-LABEL: bitcast_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: st1w { [[OP]].s }, [[PG]], [x1]
; CHECK-NEXT: ret
  %load = load volatile <8 x i32>, <8 x i32>* %a
  %cast = bitcast <8 x i32> %load to <8 x float>
  store volatile <8 x float> %cast, <8 x float>* %b
  ret void
}

define void @bitcast_v16i32(<16 x i32> *%a, <16 x float>* %b) #0 {
; CHECK-LABEL: bitcast_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: st1w { [[OP]].s }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret
  %load = load volatile <16 x i32>, <16 x i32>* %a
  %cast = bitcast <16 x i32> %load to <16 x float>
  store volatile <16 x float> %cast, <16 x float>* %b
  ret void
}

define void @bitcast_v32i32(<32 x i32> *%a, <32 x float>* %b) #0 {
; CHECK-LABEL: bitcast_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: st1w { [[OP]].s }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %load = load volatile <32 x i32>, <32 x i32>* %a
  %cast = bitcast <32 x i32> %load to <32 x float>
  store volatile <32 x float> %cast, <32 x float>* %b
  ret void
}

define void @bitcast_v64i32(<64 x i32> *%a, <64 x float>* %b) #0 {
; CHECK-LABEL: bitcast_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: st1w { [[OP]].s }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %load = load volatile <64 x i32>, <64 x i32>* %a
  %cast = bitcast <64 x i32> %load to <64 x float>
  store volatile <64 x float> %cast, <64 x float>* %b
  ret void
}

; Don't use SVE for 64-bit vectors.
define void @bitcast_v1i64(<1 x i64> *%a, <1 x double>* %b) #0 {
; CHECK-LABEL: bitcast_v1i64:
; CHECK: ldr d0, [x0]
; CHECK-NEXT: str d0, [x1]
; CHECK-NEXT: ret
  %load = load volatile <1 x i64>, <1 x i64>* %a
  %cast = bitcast <1 x i64> %load to <1 x double>
  store volatile <1 x double> %cast, <1 x double>* %b
  ret void
}

; Don't use SVE for 128-bit vectors.
define void @bitcast_v2i64(<2 x i64> *%a, <2 x double>* %b) #0 {
; CHECK-LABEL: bitcast_v2i64:
; CHECK: ldr q0, [x0]
; CHECK-NEXT: str q0, [x1]
; CHECK-NEXT: ret
  %load = load volatile <2 x i64>, <2 x i64>* %a
  %cast = bitcast <2 x i64> %load to <2 x double>
  store volatile <2 x double> %cast, <2 x double>* %b
  ret void
}

define void @bitcast_v4i64(<4 x i64> *%a, <4 x double>* %b) #0 {
; CHECK-LABEL: bitcast_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: st1d { [[OP]].d }, [[PG]], [x1]
; CHECK-NEXT: ret
  %load = load volatile <4 x i64>, <4 x i64>* %a
  %cast = bitcast <4 x i64> %load to <4 x double>
  store volatile <4 x double> %cast, <4 x double>* %b
  ret void
}

define void @bitcast_v8i64(<8 x i64> *%a, <8 x double>* %b) #0 {
; CHECK-LABEL: bitcast_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: st1d { [[OP]].d }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret
  %load = load volatile <8 x i64>, <8 x i64>* %a
  %cast = bitcast <8 x i64> %load to <8 x double>
  store volatile <8 x double> %cast, <8 x double>* %b
  ret void
}

define void @bitcast_v16i64(<16 x i64> *%a, <16 x double>* %b) #0 {
; CHECK-LABEL: bitcast_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: st1d { [[OP]].d }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %load = load volatile <16 x i64>, <16 x i64>* %a
  %cast = bitcast <16 x i64> %load to <16 x double>
  store volatile <16 x double> %cast, <16 x double>* %b
  ret void
}

define void @bitcast_v32i64(<32 x i64> *%a, <32 x double>* %b) #0 {
; CHECK-LABEL: bitcast_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: st1d { [[OP]].d }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %load = load volatile <32 x i64>, <32 x i64>* %a
  %cast = bitcast <32 x i64> %load to <32 x double>
  store volatile <32 x double> %cast, <32 x double>* %b
  ret void
}

attributes #0 = { "target-features"="+sve" }
