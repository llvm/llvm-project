; RUN: llc -aarch64-sve-vector-bits-min=128  -asm-verbose=0 < %s | FileCheck %s -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_EQ_256
; RUN: llc -aarch64-sve-vector-bits-min=384  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=512  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=640  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=768  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=896  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1024 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1152 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1280 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1408 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1536 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1664 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1792 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1920 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=2048 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_2048,VBITS_GE_1024,VBITS_GE_512

target triple = "aarch64-unknown-linux-gnu"

; Don't use SVE when its registers are no bigger than NEON.
; NO_SVE-NOT: ptrue

;
; ST1B
;

define void @masked_scatter_v2i8(<2 x i8>* %a, <2 x i8*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v2i8:
; CHECK: ldrb [[VALS_LO:w[0-9]+]], [x0]
; CHECK-NEXT: ldrb [[VALS_HI:w[0-9]+]], [x0, #1]
; CHECK-NEXT: ldr q[[PTRS:[0-9]+]], [x1]
; CHECK-NEXT: ptrue [[PG0:p[0-9]+]].d, vl2
; CHECK-NEXT: fmov s[[VALS:[0-9]+]], [[VALS_LO]]
; CHECK-NEXT: mov v[[VALS]].s[1], [[VALS_HI]]
; CHECK-NEXT: cmeq v[[CMP:[0-9]+]].2s, v[[VALS]].2s, #0
; CHECK-NEXT: ushll v[[SHL:[0-9]+]].2d, v[[CMP]].2s, #0
; CHECK-NEXT: ushll v[[SHL2:[0-9]+]].2d, v[[VALS]].2s, #0
; CHECK-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG0]]/z, z[[SHL]].d, #0
; CHECK-NEXT: st1b { z[[SHL2]].d }, [[MASK]], [z[[PTRS]].d]
; CHECK-NEXT: ret
  %vals = load <2 x i8>, <2 x i8>* %a
  %ptrs = load <2 x i8*>, <2 x i8*>* %b
  %mask = icmp eq <2 x i8> %vals, zeroinitializer
  call void @llvm.masked.scatter.v2i8(<2 x i8> %vals, <2 x i8*> %ptrs, i32 8, <2 x i1> %mask)
  ret void
}

define void @masked_scatter_v4i8(<4 x i8>* %a, <4 x i8*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v4i8:
; CHECK: ldr s[[VALS:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-NEXT: ushll [[SHL:v[0-9]+]].8h, v[[VALS]].8b, #0
; CHECK-NEXT: cmeq v[[CMP:[0-9]+]].4h, [[SHL]].4h, #0
; CHECK-NEXT: uunpklo [[UPK1:z[0-9]+]].s, z[[CMP]].h
; CHECK-NEXT: uunpklo [[UPKV1:z[0-9]+]].s, z[[VALS]].h
; CHECK-NEXT: uunpklo z[[UPK2:[0-9]+]].d, [[UPK1]].s
; CHECK-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG]]/z, z[[UPK2]].d, #0
; CHECK-NEXT: uunpklo [[UPKV2:z[0-9]+]].d, [[UPKV1]].s
; CHECK-NEXT: st1b { [[UPKV2]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; CHECK-NEXT: ret
  %vals = load <4 x i8>, <4 x i8>* %a
  %ptrs = load <4 x i8*>, <4 x i8*>* %b
  %mask = icmp eq <4 x i8> %vals, zeroinitializer
  call void @llvm.masked.scatter.v4i8(<4 x i8> %vals, <4 x i8*> %ptrs, i32 8, <4 x i1> %mask)
  ret void
}

define void @masked_scatter_v8i8(<8 x i8>* %a, <8 x i8*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v8i8:
; VBITS_GE_512: ldr d[[VALS:[0-9]+]], [x0]
; VBITS_GE_512-NEXT: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: cmeq v[[CMP:[0-9]+]].8b, v[[VALS]].8b, #0
; VBITS_GE_512-NEXT: uunpklo [[UPK1:z[0-9]+]].h, z[[CMP]].b
; VBITS_GE_512-NEXT: uunpklo [[UPKV1:z[0-9]+]].h, z[[VALS]].b
; VBITS_GE_512-NEXT: uunpklo [[UPK2:z[0-9]+]].s, [[UPK1]].h
; VBITS_GE_512-NEXT: uunpklo [[UPKV2:z[0-9]+]].s, [[UPKV1]].h
; VBITS_GE_512-NEXT: uunpklo [[UPK3:z[0-9]+]].d, [[UPK2]].s
; VBITS_GE_512-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG]]/z, [[UPK3]].d, #0
; VBITS_GE_512-NEXT: uunpklo [[UPKV3:z[0-9]+]].d, [[UPKV2]].s
; VBITS_GE_512-NEXT: st1b { [[UPKV3]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ldr d[[VALS:[0-9]+]], [x0]
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: cmeq [[ZMSK:v[0-9]+]].8b, v[[VALS]].8b, #0
; VBITS_EQ_256-DAG: zip1 v[[VAL_LO:[0-9]+]].8b, [[ZMSK]].8b, v[[VALS]].8b
; VBITS_EQ_256-DAG: zip2 v[[VAL_HI:[0-9]+]].8b, [[ZMSK]].8b, v[[VALS]].8b
; VBITS_EQ_256-DAG: shl [[SHL_LO:v[0-9]+]].4h, v[[VAL_LO]].4h, #8
; VBITS_EQ_256-DAG: shl [[SHL_HI:v[0-9]+]].4h, v[[VAL_HI]].4h, #8
; VBITS_EQ_256-DAG: ld1d { [[PTRS_LO:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_EQ_256-DAG: ld1d { [[PTRS_HI:z[0-9]+]].d }, [[PG]]/z, [x1, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-DAG: sshr v[[SSHR_LO:[0-9]+]].4h, [[SHL_LO]].4h, #8
; VBITS_EQ_256-DAG: sshr v[[SSHR_HI:[0-9]+]].4h, [[SHL_HI]].4h, #8
; VBITS_EQ_256-DAG: uunpklo [[UPK1_LO:z[0-9]+]].s, z[[VAL_LO]].h
; VBITS_EQ_256-DAG: uunpklo [[UPK1_HI:z[0-9]+]].s, z[[VAL_HI]].h
; VBITS_EQ_256-DAG: uunpklo z[[UPK2_LO:[0-9]+]].d, [[UPK1_LO]].s
; VBITS_EQ_256-DAG: uunpklo z[[UPK2_HI:[0-9]+]].d, [[UPK1_HI]].s
; VBITS_EQ_256-DAG: cmpne [[MASK_LO:p[0-9]+]].d, [[PG]]/z, z[[UPK2_LO]].d, #0
; VBITS_EQ_256-DAG: cmpne [[MASK_HI:p[0-9]+]].d, [[PG]]/z, z[[UPK2_HI]].d, #0
; VBITS_EQ_256-DAG: zip1 v[[VALS2_LO:[0-9]+]].8b, v[[VALS]].8b, v[[VALS]].8b
; VBITS_EQ_256-DAG: zip2 v[[VALS2_HI:[0-9]+]].8b, v[[VALS]].8b, v[[VALS]].8b
; VBITS_EQ_256-DAG: uunpklo [[UPK1_LO:z[0-9]+]].s, z[[VALS2_LO]].h
; VBITS_EQ_256-DAG: uunpklo [[UPK1_HI:z[0-9]+]].s, z[[VALS2_HI]].h
; VBITS_EQ_256-DAG: uunpklo [[UPK2_LO:z[0-9]+]].d, [[UPK1_LO]].s
; VBITS_EQ_256-DAG: uunpklo [[UPK2_HI:z[0-9]+]].d, [[UPK1_HI]].s
; VBITS_EQ_256-DAG: st1b { [[UPK2_LO]].d }, [[MASK_LO]], {{\[}}[[PTRS_LO]].d]
; VBITS_EQ_256-DAG: st1b { [[UPK2_HI]].d }, [[MASK_HI]], {{\[}}[[PTRS_HI]].d]
; VBITS_EQ_256-NEXT: ret
  %vals = load <8 x i8>, <8 x i8>* %a
  %ptrs = load <8 x i8*>, <8 x i8*>* %b
  %mask = icmp eq <8 x i8> %vals, zeroinitializer
  call void @llvm.masked.scatter.v8i8(<8 x i8> %vals, <8 x i8*> %ptrs, i32 8, <8 x i1> %mask)
  ret void
}

define void @masked_scatter_v16i8(<16 x i8>* %a, <16 x i8*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v16i8:
; VBITS_GE_1024: ldr q[[VALS:[0-9]+]], [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: cmeq v[[CMP:[0-9]+]].16b, v[[VALS]].16b, #0
; VBITS_GE_1024-NEXT: uunpklo [[UPK1:z[0-9]+]].h, z[[CMP]].b
; VBITS_GE_1024-NEXT: uunpklo [[UPKV1:z[0-9]+]].h, z[[VALS]].b
; VBITS_GE_1024-NEXT: uunpklo [[UPK2:z[0-9]+]].s, [[UPK1]].h
; VBITS_GE_1024-NEXT: uunpklo [[UPKV2:z[0-9]+]].s, [[UPKV1]].h
; VBITS_GE_1024-NEXT: uunpklo [[UPK3:z[0-9]+]].d, [[UPK2]].s
; VBITS_GE_1024-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG]]/z, [[UPK3]].d, #0
; VBITS_GE_1024-NEXT: uunpklo [[UPKV3:z[0-9]+]].d, [[UPKV2]].s
; VBITS_GE_1024-NEXT: st1b { [[UPKV3]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; VBITS_GE_1024-NEXT: ret
  %vals = load <16 x i8>, <16 x i8>* %a
  %ptrs = load <16 x i8*>, <16 x i8*>* %b
  %mask = icmp eq <16 x i8> %vals, zeroinitializer
  call void @llvm.masked.scatter.v16i8(<16 x i8> %vals, <16 x i8*> %ptrs, i32 8, <16 x i1> %mask)
  ret void
}

define void @masked_scatter_v32i8(<32 x i8>* %a, <32 x i8*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v32i8:
; VBITS_GE_2048: ptrue [[PG0:p[0-9]+]].b, vl32
; VBITS_GE_2048-NEXT: ld1b { [[VALS:z[0-9]+]].b }, [[PG0]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG1:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: cmpeq [[CMP:p[0-9]+]].b, [[PG0]]/z, [[VALS]].b, #0
; VBITS_GE_2048-NEXT: mov [[MONE:z[0-9]+]].b, [[PG0]]/z, #-1
; VBITS_GE_2048-NEXT: uunpklo [[UPK1:z[0-9]+]].h, [[MONE]].b
; VBITS_GE_2048-NEXT: uunpklo [[UPKV1:z[0-9]+]].h, [[VALS]].b
; VBITS_GE_2048-NEXT: uunpklo [[UPK2:z[0-9]+]].s, [[UPK1]].h
; VBITS_GE_2048-NEXT: uunpklo [[UPKV2:z[0-9]+]].s, [[UPKV1]].h
; VBITS_GE_2048-NEXT: uunpklo [[UPK3:z[0-9]+]].d, [[UPK2]].s
; VBITS_GE_2048-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG1]]/z, [[UPK3]].d, #0
; VBITS_GE_2048-NEXT: uunpklo [[UPKV3:z[0-9]+]].d, [[UPKV2]].s
; VBITS_GE_2048-NEXT: st1b { [[UPKV3]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; VBITS_GE_2048-NEXT: ret
  %vals = load <32 x i8>, <32 x i8>* %a
  %ptrs = load <32 x i8*>, <32 x i8*>* %b
  %mask = icmp eq <32 x i8> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32i8(<32 x i8> %vals, <32 x i8*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

;
; ST1H
;

define void @masked_scatter_v2i16(<2 x i16>* %a, <2 x i16*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v2i16:
; CHECK: ldrh [[VALS_LO:w[0-9]+]], [x0]
; CHECK-NEXT: ldrh [[VALS_HI:w[0-9]+]], [x0, #2]
; CHECK-NEXT: ldr q[[PTRS:[0-9]+]], [x1]
; CHECK-NEXT: ptrue [[PG0:p[0-9]+]].d, vl2
; CHECK-NEXT: fmov s[[VALS:[0-9]+]], [[VALS_LO]]
; CHECK-NEXT: mov v[[VALS]].s[1], [[VALS_HI]]
; CHECK-NEXT: cmeq v[[CMP:[0-9]+]].2s, v[[VALS]].2s, #0
; CHECK-NEXT: ushll v[[SHL:[0-9]+]].2d, v[[CMP]].2s, #0
; CHECK-NEXT: ushll v[[SHL2:[0-9]+]].2d, v[[VALS]].2s, #0
; CHECK-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG0]]/z, z[[SHL]].d, #0
; CHECK-NEXT: st1h { z[[SHL2]].d }, [[MASK]], [z[[PTRS]].d]
; CHECK-NEXT: ret
  %vals = load <2 x i16>, <2 x i16>* %a
  %ptrs = load <2 x i16*>, <2 x i16*>* %b
  %mask = icmp eq <2 x i16> %vals, zeroinitializer
  call void @llvm.masked.scatter.v2i16(<2 x i16> %vals, <2 x i16*> %ptrs, i32 8, <2 x i1> %mask)
  ret void
}

define void @masked_scatter_v4i16(<4 x i16>* %a, <4 x i16*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v4i16:
; CHECK: ldr d[[VALS:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-NEXT: cmeq v[[CMP:[0-9]+]].4h, v[[VALS]].4h, #0
; CHECK-NEXT: uunpklo [[UPK1:z[0-9]+]].s, z[[CMP]].h
; CHECK-NEXT: uunpklo [[UPKV1:z[0-9]+]].s, z[[VALS]].h
; CHECK-NEXT: uunpklo z[[UPK2:[0-9]+]].d, [[UPK1]].s
; CHECK-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG]]/z, z[[UPK2]].d, #0
; CHECK-NEXT: uunpklo [[UPKV2:z[0-9]+]].d, [[UPKV1]].s
; CHECK-NEXT: st1h { [[UPKV2]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; CHECK-NEXT: ret
  %vals = load <4 x i16>, <4 x i16>* %a
  %ptrs = load <4 x i16*>, <4 x i16*>* %b
  %mask = icmp eq <4 x i16> %vals, zeroinitializer
  call void @llvm.masked.scatter.v4i16(<4 x i16> %vals, <4 x i16*> %ptrs, i32 8, <4 x i1> %mask)
  ret void
}

define void @masked_scatter_v8i16(<8 x i16>* %a, <8 x i16*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v8i16:
; VBITS_GE_512: ldr q[[VALS:[0-9]+]], [x0]
; VBITS_GE_512-NEXT: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: cmeq v[[CMP:[0-9]+]].8h, v[[VALS]].8h, #0
; VBITS_GE_512-NEXT: uunpklo [[UPK1:z[0-9]+]].s, z[[CMP]].h
; VBITS_GE_512-NEXT: uunpklo [[UPKV1:z[0-9]+]].s, z[[VALS]].h
; VBITS_GE_512-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; VBITS_GE_512-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG]]/z, [[UPK2]].d, #0
; VBITS_GE_512-NEXT: uunpklo [[UPKV2:z[0-9]+]].d, [[UPKV1]].s
; VBITS_GE_512-NEXT: st1h { [[UPKV2]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ldr q[[VALS:[0-9]+]], [x0]
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: ld1d { [[PTRS_LO:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_EQ_256-DAG: ld1d { [[PTRS_HI:z[0-9]+]].d }, [[PG]]/z, [x1, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-DAG: cmeq v[[ZMSK:[0-9]+]].8h, v[[VALS]].8h, #0
; VBITS_EQ_256-DAG: ext v[[EXT:[0-9]+]].16b, v[[VALS]].16b, v[[VALS]].16b, #8
; VBITS_EQ_256-DAG: ext v[[ZEXT:[0-9]+]].16b, v[[ZMSK]].16b, v[[ZMSK]].16b, #8
; VBITS_EQ_256-DAG: uunpklo [[UPK1_LO:z[0-9]+]].s, z[[ZMSK]].h
; VBITS_EQ_256-DAG: uunpklo [[UPK1_HI:z[0-9]+]].s, z[[ZEXT]].h
; VBITS_EQ_256-DAG: uunpklo [[UPK2_LO:z[0-9]+]].d, [[UPK1_LO]].s
; VBITS_EQ_256-DAG: uunpklo [[UPK2_HI:z[0-9]+]].d, [[UPK1_HI]].s
; VBITS_EQ_256-DAG: cmpne [[MASK_LO:p[0-9]+]].d, [[PG]]/z, [[UPK2_LO]].d, #0
; VBITS_EQ_256-DAG: cmpne [[MASK_HI:p[0-9]+]].d, [[PG]]/z, [[UPK2_HI]].d, #0
; VBITS_EQ_256-DAG: uunpklo [[UPK1_LO:z[0-9]+]].s, z[[VALS]].h
; VBITS_EQ_256-DAG: uunpklo [[UPK1_HI:z[0-9]+]].s, z[[EXT]].h
; VBITS_EQ_256-DAG: uunpklo [[UPK2_LO:z[0-9]+]].d, [[UPK1_LO]].s
; VBITS_EQ_256-DAG: uunpklo [[UPK2_HI:z[0-9]+]].d, [[UPK1_HI]].s
; VBITS_EQ_256-DAG: st1h { [[UPK2_LO]].d }, [[MASK_LO]], {{\[}}[[PTRS_LO]].d]
; VBITS_EQ_256-DAG: st1h { [[UPK2_HI]].d }, [[MASK_HI]], {{\[}}[[PTRS_HI]].d]
; VBITS_EQ_256-NEXT: ret
  %vals = load <8 x i16>, <8 x i16>* %a
  %ptrs = load <8 x i16*>, <8 x i16*>* %b
  %mask = icmp eq <8 x i16> %vals, zeroinitializer
  call void @llvm.masked.scatter.v8i16(<8 x i16> %vals, <8 x i16*> %ptrs, i32 8, <8 x i1> %mask)
  ret void
}

define void @masked_scatter_v16i16(<16 x i16>* %a, <16 x i16*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v16i16:
; VBITS_GE_1024: ptrue [[PG0:p[0-9]+]].h, vl16
; VBITS_GE_1024-NEXT: ld1h { [[VALS:z[0-9]+]].h }, [[PG0]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG1:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: cmpeq [[CMP:p[0-9]+]].h, [[PG0]]/z, [[VALS]].h, #0
; VBITS_GE_1024-NEXT: mov [[MONE:z[0-9]+]].h, [[CMP]]/z, #-1
; VBITS_GE_1024-NEXT: uunpklo [[UPK1:z[0-9]+]].s, [[MONE]].h
; VBITS_GE_1024-NEXT: uunpklo [[UPKV1:z[0-9]+]].s, [[VALS]].h
; VBITS_GE_1024-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; VBITS_GE_1024-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG1]]/z, [[UPK2]].d, #0
; VBITS_GE_1024-NEXT: uunpklo [[UPKV2:z[0-9]+]].d, [[UPKV1]].s
; VBITS_GE_1024-NEXT: st1h { [[UPKV2]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; VBITS_GE_1024-NEXT: ret
  %vals = load <16 x i16>, <16 x i16>* %a
  %ptrs = load <16 x i16*>, <16 x i16*>* %b
  %mask = icmp eq <16 x i16> %vals, zeroinitializer
  call void @llvm.masked.scatter.v16i16(<16 x i16> %vals, <16 x i16*> %ptrs, i32 8, <16 x i1> %mask)
  ret void
}

define void @masked_scatter_v32i16(<32 x i16>* %a, <32 x i16*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v32i16:
; VBITS_GE_2048: ptrue [[PG0:p[0-9]+]].h, vl32
; VBITS_GE_2048-NEXT: ld1h { [[VALS:z[0-9]+]].h }, [[PG0]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG1:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: cmpeq [[CMP:p[0-9]+]].h, [[PG0]]/z, [[VALS]].h, #0
; VBITS_GE_2048-NEXT: mov [[MONE:z[0-9]+]].h, [[CMP]]/z, #-1
; VBITS_GE_2048-NEXT: uunpklo [[UPK1:z[0-9]+]].s, [[MONE]].h
; VBITS_GE_2048-NEXT: uunpklo [[UPKV1:z[0-9]+]].s, [[VALS]].h
; VBITS_GE_2048-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; VBITS_GE_2048-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG1]]/z, [[UPK2]].d, #0
; VBITS_GE_2048-NEXT: uunpklo [[UPKV2:z[0-9]+]].d, [[UPKV1]].s
; VBITS_GE_2048-NEXT: st1h { [[UPKV2]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; VBITS_GE_2048-NEXT: ret
  %vals = load <32 x i16>, <32 x i16>* %a
  %ptrs = load <32 x i16*>, <32 x i16*>* %b
  %mask = icmp eq <32 x i16> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32i16(<32 x i16> %vals, <32 x i16*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

;
; ST1W
;

define void @masked_scatter_v2i32(<2 x i32>* %a, <2 x i32*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v2i32:
; CHECK: ldr d[[VALS:[0-9]+]], [x0]
; CHECK-NEXT: ldr q[[PTRS:[0-9]+]], [x1]
; CHECK-NEXT: ptrue [[PG0:p[0-9]+]].d, vl2
; CHECK-NEXT: cmeq v[[CMP:[0-9]+]].2s, v[[VALS]].2s, #0
; CHECK-NEXT: ushll v[[SHL:[0-9]+]].2d, v[[CMP]].2s, #0
; CHECK-NEXT: ushll v[[SHL2:[0-9]+]].2d, v[[VALS]].2s, #0
; CHECK-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG0]]/z, z[[SHL]].d, #0
; CHECK-NEXT: st1w { z[[SHL2]].d }, [[MASK]], [z[[PTRS]].d]
; CHECK-NEXT: ret
  %vals = load <2 x i32>, <2 x i32>* %a
  %ptrs = load <2 x i32*>, <2 x i32*>* %b
  %mask = icmp eq <2 x i32> %vals, zeroinitializer
  call void @llvm.masked.scatter.v2i32(<2 x i32> %vals, <2 x i32*> %ptrs, i32 8, <2 x i1> %mask)
  ret void
}

define void @masked_scatter_v4i32(<4 x i32>* %a, <4 x i32*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v4i32:
; CHECK: ldr q[[VALS:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-NEXT: cmeq v[[CMP:[0-9]+]].4s, v[[VALS]].4s, #0
; CHECK-NEXT: uunpklo [[UPK:z[0-9]+]].d, z[[CMP]].s
; CHECK-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG]]/z, [[UPK]].d, #0
; CHECK-NEXT: uunpklo [[UPKV:z[0-9]+]].d, z[[VALS]].s
; CHECK-NEXT: st1w { [[UPKV]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; CHECK-NEXT: ret
  %vals = load <4 x i32>, <4 x i32>* %a
  %ptrs = load <4 x i32*>, <4 x i32*>* %b
  %mask = icmp eq <4 x i32> %vals, zeroinitializer
  call void @llvm.masked.scatter.v4i32(<4 x i32> %vals, <4 x i32*> %ptrs, i32 8, <4 x i1> %mask)
  ret void
}

define void @masked_scatter_v8i32(<8 x i32>* %a, <8 x i32*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v8i32:
; VBITS_GE_512: ptrue [[PG0:p[0-9]+]].s, vl8
; VBITS_GE_512-NEXT: ld1w { [[VALS:z[0-9]+]].s }, [[PG0]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG1:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_512-NEXT: cmpeq [[CMP:p[0-9]+]].s, [[PG0]]/z, [[VALS]].s, #0
; VBITS_GE_512-NEXT: mov [[MONE:z[0-9]+]].s, [[CMP]]/z, #-1
; VBITS_GE_512-NEXT: uunpklo [[UPK:z[0-9]+]].d, [[MONE]].s
; VBITS_GE_512-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG1]]/z, [[UPK]].d, #0
; VBITS_GE_512-NEXT: uunpklo [[UPKV:z[0-9]+]].d, [[VALS]].s
; VBITS_GE_512-NEXT: st1w { [[UPKV]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG0:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: ld1w { [[VALS:z[0-9]+]].s }, [[PG0]]/z, [x0]
; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: ld1d { [[PTRS_LO:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_EQ_256-DAG: ld1d { [[PTRS_HI:z[0-9]+]].d }, [[PG1]]/z, [x1, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-DAG: cmpeq [[MASK:p[0-9]+]].s, [[PG0]]/z, [[VALS]].s, #0
; VBITS_EQ_256-DAG: add x8, sp, #32
; VBITS_EQ_256-DAG: mov x9, sp
; VBITS_EQ_256-DAG: mov [[MONE:z[0-9]+]].s, [[MASK]]/z, #-1
; VBITS_EQ_256-DAG: st1w  { [[MONE]].s }, [[PG0]], [x8]
; VBITS_EQ_256-DAG: st1w  { [[VALS]].s }, [[PG0]], [x9]
; VBITS_EQ_256-DAG: ldr q[[CMP_LO:[0-9]+]], [sp, #32]
; VBITS_EQ_256-DAG: ldr q[[VAL_LO:[0-9]+]], [sp]
; VBITS_EQ_256-DAG: uunpklo [[UPKC_LO:z[0-9]+]].d, z[[CMP_LO]].s
; VBITS_EQ_256-DAG: cmpne [[MASK_LO:p[0-9]+]].d, [[PG1]]/z, [[UPKC_LO]].d, #0
; VBITS_EQ_256-DAG: uunpklo [[UPK1_LO:z[0-9]+]].d, z[[VAL_LO]].s
; VBITS_EQ_256-DAG: st1w { [[UPK1_LO]].d }, [[MASK_LO]], {{\[}}[[PTRS_LO]].d]
; VBITS_EQ_256-DAG: ldr q[[CMP_HI:[0-9]+]], [sp, #48]
; VBITS_EQ_256-DAG: ldr q[[VAL_HI:[0-9]+]], [sp, #16]
; VBITS_EQ_256-DAG: uunpklo [[UPKC_HI:z[0-9]+]].d, z[[CMP_HI]].s
; VBITS_EQ_256-DAG: cmpne [[MASK_HI:p[0-9]+]].d, [[PG1]]/z, [[UPKC_HI]].d, #0
; VBITS_EQ_256-DAG: uunpklo [[UPK1_HI:z[0-9]+]].d, z[[VAL_HI]].s
; VBITS_EQ_256-DAG: st1w { [[UPK1_HI]].d }, [[MASK_HI]], {{\[}}[[PTRS_HI]].d]
  %vals = load <8 x i32>, <8 x i32>* %a
  %ptrs = load <8 x i32*>, <8 x i32*>* %b
  %mask = icmp eq <8 x i32> %vals, zeroinitializer
  call void @llvm.masked.scatter.v8i32(<8 x i32> %vals, <8 x i32*> %ptrs, i32 8, <8 x i1> %mask)
  ret void
}

define void @masked_scatter_v16i32(<16 x i32>* %a, <16 x i32*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v16i32:
; VBITS_GE_1024: ptrue [[PG0:p[0-9]+]].s, vl16
; VBITS_GE_1024-NEXT: ld1w { [[VALS:z[0-9]+]].s }, [[PG0]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG1:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: cmpeq [[CMP:p[0-9]+]].s, [[PG0]]/z, [[VALS]].s, #0
; VBITS_GE_1024-NEXT: mov [[MONE:z[0-9]+]].s, [[CMP]]/z, #-1
; VBITS_GE_1024-NEXT: uunpklo [[UPK:z[0-9]+]].d, [[MONE]].s
; VBITS_GE_1024-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG1]]/z, [[UPK]].d, #0
; VBITS_GE_1024-NEXT: uunpklo [[UPKV:z[0-9]+]].d, [[VALS]].s
; VBITS_GE_1024-NEXT: st1w { [[UPKV]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; VBITS_GE_1024-NEXT: ret
  %vals = load <16 x i32>, <16 x i32>* %a
  %ptrs = load <16 x i32*>, <16 x i32*>* %b
  %mask = icmp eq <16 x i32> %vals, zeroinitializer
  call void @llvm.masked.scatter.v16i32(<16 x i32> %vals, <16 x i32*> %ptrs, i32 8, <16 x i1> %mask)
  ret void
}

define void @masked_scatter_v32i32(<32 x i32>* %a, <32 x i32*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v32i32:
; VBITS_GE_2048: ptrue [[PG0:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: ld1w { [[VALS:z[0-9]+]].s }, [[PG0]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG1:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: cmpeq [[CMP:p[0-9]+]].s, [[PG0]]/z, [[VALS]].s, #0
; VBITS_GE_2048-NEXT: mov [[MONE:z[0-9]+]].s, [[CMP]]/z, #-1
; VBITS_GE_2048-NEXT: uunpklo [[UPK:z[0-9]+]].d, [[MONE]].s
; VBITS_GE_2048-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG1]]/z, [[UPK]].d, #0
; VBITS_GE_2048-NEXT: uunpklo [[UPKV:z[0-9]+]].d, [[VALS]].s
; VBITS_GE_2048-NEXT: st1w { [[UPKV]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; VBITS_GE_2048-NEXT: ret
  %vals = load <32 x i32>, <32 x i32>* %a
  %ptrs = load <32 x i32*>, <32 x i32*>* %b
  %mask = icmp eq <32 x i32> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32i32(<32 x i32> %vals, <32 x i32*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

;
; ST1D
;

; Scalarize 1 x i64 scatters
define void @masked_scatter_v1i64(<1 x i64>* %a, <1 x i64*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v1i64:
; CHECK-NOT: ptrue
  %vals = load <1 x i64>, <1 x i64>* %a
  %ptrs = load <1 x i64*>, <1 x i64*>* %b
  %mask = icmp eq <1 x i64> %vals, zeroinitializer
  call void @llvm.masked.scatter.v1i64(<1 x i64> %vals, <1 x i64*> %ptrs, i32 8, <1 x i1> %mask)
  ret void
}

define void @masked_scatter_v2i64(<2 x i64>* %a, <2 x i64*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v2i64:
; CHECK: ldr q[[VALS:[0-9]+]], [x0]
; CHECK-NEXT: ldr q[[PTRS:[0-9]+]], [x1]
; CHECK-NEXT: ptrue [[PG0:p[0-9]+]].d, vl2
; CHECK-NEXT: cmeq v[[CMP:[0-9]+]].2d, v[[VALS]].2d, #0
; CHECK-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG0]]/z, z[[CMP]].d, #0
; CHECK-NEXT: st1d { z[[VALS]].d }, [[MASK]], [z[[PTRS]].d]
; CHECK-NEXT: ret
  %vals = load <2 x i64>, <2 x i64>* %a
  %ptrs = load <2 x i64*>, <2 x i64*>* %b
  %mask = icmp eq <2 x i64> %vals, zeroinitializer
  call void @llvm.masked.scatter.v2i64(<2 x i64> %vals, <2 x i64*> %ptrs, i32 8, <2 x i1> %mask)
  ret void
}

define void @masked_scatter_v4i64(<4 x i64>* %a, <4 x i64*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v4i64:
; CHECK: ptrue [[PG0:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[VALS:z[0-9]+]].d }, [[PG0]]/z, [x0]
; CHECK-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG0]]/z, [x1]
; CHECK-NEXT: cmpeq [[MASK:p[0-9]+]].d, [[PG0]]/z, [[VALS]].d, #0
; CHECK-NEXT: st1d { [[VALS]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; CHECK-NEXT: ret
  %vals = load <4 x i64>, <4 x i64>* %a
  %ptrs = load <4 x i64*>, <4 x i64*>* %b
  %mask = icmp eq <4 x i64> %vals, zeroinitializer
  call void @llvm.masked.scatter.v4i64(<4 x i64> %vals, <4 x i64*> %ptrs, i32 8, <4 x i1> %mask)
  ret void
}

define void @masked_scatter_v8i64(<8 x i64>* %a, <8 x i64*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v8i64:
; VBITS_GE_512: ptrue [[PG0:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[VALS:z[0-9]+]].d }, [[PG0]]/z, [x0]
; VBITS_GE_512-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG0]]/z, [x1]
; VBITS_GE_512-NEXT: cmpeq [[MASK:p[0-9]+]].d, [[PG0]]/z, [[VALS]].d, #0
; VBITS_GE_512-NEXT: st1d { [[VALS]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG0:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: ld1d { [[VALS_LO:z[0-9]+]].d }, [[PG0]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[VALS_HI:z[0-9]+]].d }, [[PG0]]/z, [x0, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-DAG: ld1d { [[PTRS_LO:z[0-9]+]].d }, [[PG0]]/z, [x1]
; VBITS_EQ_256-DAG: ld1d { [[PTRS_HI:z[0-9]+]].d }, [[PG0]]/z, [x1, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-DAG: cmpeq [[MASK_LO:p[0-9]+]].d, [[PG0]]/z, [[VALS_LO]].d, #0
; VBITS_EQ_256-DAG: cmpeq [[MASK_HI:p[0-9]+]].d, [[PG0]]/z, [[VALS_HI]].d, #0
; VBITS_EQ_256-DAG: st1d { [[VALS_LO]].d }, [[MASK_LO]], {{\[}}[[PTRS_LO]].d]
; VBITS_EQ_256-DAG: st1d { [[VALS_HI]].d }, [[MASK_HI]], {{\[}}[[PTRS_HI]].d]
; VBITS_EQ_256-NEXT: ret
  %vals = load <8 x i64>, <8 x i64>* %a
  %ptrs = load <8 x i64*>, <8 x i64*>* %b
  %mask = icmp eq <8 x i64> %vals, zeroinitializer
  call void @llvm.masked.scatter.v8i64(<8 x i64> %vals, <8 x i64*> %ptrs, i32 8, <8 x i1> %mask)
  ret void
}

define void @masked_scatter_v16i64(<16 x i64>* %a, <16 x i64*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v16i64:
; VBITS_GE_1024: ptrue [[PG0:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[VALS:z[0-9]+]].d }, [[PG0]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG0]]/z, [x1]
; VBITS_GE_1024-NEXT: cmpeq [[MASK:p[0-9]+]].d, [[PG0]]/z, [[VALS]].d, #0
; VBITS_GE_1024-NEXT: st1d { [[VALS]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; VBITS_GE_1024-NEXT: ret
  %vals = load <16 x i64>, <16 x i64>* %a
  %ptrs = load <16 x i64*>, <16 x i64*>* %b
  %mask = icmp eq <16 x i64> %vals, zeroinitializer
  call void @llvm.masked.scatter.v16i64(<16 x i64> %vals, <16 x i64*> %ptrs, i32 8, <16 x i1> %mask)
  ret void
}

define void @masked_scatter_v32i64(<32 x i64>* %a, <32 x i64*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v32i64:
; VBITS_GE_2048: ptrue [[PG0:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[VALS:z[0-9]+]].d }, [[PG0]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG0]]/z, [x1]
; VBITS_GE_2048-NEXT: cmpeq [[MASK:p[0-9]+]].d, [[PG0]]/z, [[VALS]].d, #0
; VBITS_GE_2048-NEXT: st1d { [[VALS]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; VBITS_GE_2048-NEXT: ret
  %vals = load <32 x i64>, <32 x i64>* %a
  %ptrs = load <32 x i64*>, <32 x i64*>* %b
  %mask = icmp eq <32 x i64> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32i64(<32 x i64> %vals, <32 x i64*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

;
; ST1H (float)
;

define void @masked_scatter_v2f16(<2 x half>* %a, <2 x half*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v2f16:
; CHECK: ldr s[[VALS:[0-9]+]], [x0]
; CHECK-NEXT: movi d2, #0000000000000000
; CHECK-NEXT: ldr q[[PTRS:[0-9]+]], [x1]
; CHECK-NEXT: ptrue [[PG0:p[0-9]+]].d, vl4
; CHECK-NEXT: fcmeq v[[CMP:[0-9]+]].4h, v[[VALS]].4h, #0.0
; CHECK-NEXT: umov w8, v[[CMP]].h[0]
; CHECK-NEXT: umov w9, v[[CMP]].h[1]
; CHECK-NEXT: fmov s[[CMP]], w8
; CHECK-NEXT: mov v[[CMP]].s[1], w9
; CHECK-NEXT: shl v[[CMP]].2s, v[[CMP]].2s, #16
; CHECK-NEXT: sshr v[[CMP]].2s, v[[CMP]].2s, #16
; CHECK-NEXT: fmov w9, s[[CMP]]
; CHECK-NEXT: mov w8, v[[CMP]].s[1]
; CHECK-NEXT: mov v[[NCMP:[0-9]+]].h[0], w9
; CHECK-NEXT: mov v[[NCMP]].h[1], w8
; CHECK-NEXT: shl v[[NCMP]].4h, v[[NCMP]].4h, #15
; CHECK-NEXT: sshr v[[NCMP]].4h, v[[NCMP]].4h, #15
; CHECK-NEXT: uunpklo [[UPK1:z[0-9]+]].s, z[[NCMP]].h
; CHECK-NEXT: uunpklo [[UPKV1:z[0-9]+]].s, z[[VALS]].h
; CHECK-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; CHECK-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG0]]/z, [[UPK2]].d, #0
; CHECK-NEXT: uunpklo [[UPKV2:z[0-9]+]].d, [[UPKV1]].s
; CHECK-NEXT: st1h { [[UPKV2]].d }, [[MASK]], [z[[PTRS]].d]
; CHECK-NEXT: ret
  %vals = load <2 x half>, <2 x half>* %a
  %ptrs = load <2 x half*>, <2 x half*>* %b
  %mask = fcmp oeq <2 x half> %vals, zeroinitializer
  call void @llvm.masked.scatter.v2f16(<2 x half> %vals, <2 x half*> %ptrs, i32 8, <2 x i1> %mask)
  ret void
}

define void @masked_scatter_v4f16(<4 x half>* %a, <4 x half*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v4f16:
; CHECK: ldr d[[VALS:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-NEXT: fcmeq v[[CMP:[0-9]+]].4h, v[[VALS]].4h, #0
; CHECK-NEXT: uunpklo [[UPK1:z[0-9]+]].s, z[[CMP]].h
; CHECK-NEXT: uunpklo [[UPKV1:z[0-9]+]].s, z[[VALS]].h
; CHECK-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; CHECK-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG]]/z, [[UPK2]].d, #0
; CHECK-NEXT: uunpklo [[UPKV2:z[0-9]+]].d, [[UPKV1]].s
; CHECK-NEXT: st1h { [[UPKV2]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; CHECK-NEXT: ret
  %vals = load <4 x half>, <4 x half>* %a
  %ptrs = load <4 x half*>, <4 x half*>* %b
  %mask = fcmp oeq <4 x half> %vals, zeroinitializer
  call void @llvm.masked.scatter.v4f16(<4 x half> %vals, <4 x half*> %ptrs, i32 8, <4 x i1> %mask)
  ret void
}

define void @masked_scatter_v8f16(<8 x half>* %a, <8 x half*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v8f16:
; VBITS_GE_512: ldr q[[VALS:[0-9]+]], [x0]
; VBITS_GE_512-NEXT: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: fcmeq v[[CMP:[0-9]+]].8h, v[[VALS]].8h, #0
; VBITS_GE_512-NEXT: uunpklo [[UPK1:z[0-9]+]].s, z[[CMP]].h
; VBITS_GE_512-NEXT: uunpklo [[UPKV1:z[0-9]+]].s, z[[VALS]].h
; VBITS_GE_512-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; VBITS_GE_512-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG]]/z, [[UPK2]].d, #0
; VBITS_GE_512-NEXT: uunpklo [[UPKV2:z[0-9]+]].d, [[UPKV1]].s
; VBITS_GE_512-NEXT: st1h { [[UPKV2]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; VBITS_GE_512-NEXT: ret
  %vals = load <8 x half>, <8 x half>* %a
  %ptrs = load <8 x half*>, <8 x half*>* %b
  %mask = fcmp oeq <8 x half> %vals, zeroinitializer
  call void @llvm.masked.scatter.v8f16(<8 x half> %vals, <8 x half*> %ptrs, i32 8, <8 x i1> %mask)
  ret void
}

define void @masked_scatter_v16f16(<16 x half>* %a, <16 x half*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v16f16:
; VBITS_GE_1024: ptrue [[PG0:p[0-9]+]].h, vl16
; VBITS_GE_1024-NEXT: ld1h { [[VALS:z[0-9]+]].h }, [[PG0]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG1:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: fcmeq [[CMP:p[0-9]+]].h, [[PG0]]/z, [[VALS]].h, #0.0
; VBITS_GE_1024-NEXT: mov [[MONE:z[0-9]+]].h, [[CMP]]/z, #-1
; VBITS_GE_1024-NEXT: uunpklo [[UPK1:z[0-9]+]].s, [[MONE]].h
; VBITS_GE_1024-NEXT: uunpklo [[UPKV1:z[0-9]+]].s, [[VALS]].h
; VBITS_GE_1024-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; VBITS_GE_1024-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG1]]/z, [[UPK2]].d, #0
; VBITS_GE_1024-NEXT: uunpklo [[UPKV2:z[0-9]+]].d, [[UPKV1]].s
; VBITS_GE_1024-NEXT: st1h { [[UPKV2]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; VBITS_GE_1024-NEXT: ret
  %vals = load <16 x half>, <16 x half>* %a
  %ptrs = load <16 x half*>, <16 x half*>* %b
  %mask = fcmp oeq <16 x half> %vals, zeroinitializer
  call void @llvm.masked.scatter.v16f16(<16 x half> %vals, <16 x half*> %ptrs, i32 8, <16 x i1> %mask)
  ret void
}

define void @masked_scatter_v32f16(<32 x half>* %a, <32 x half*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v32f16:
; VBITS_GE_2048: ptrue [[PG0:p[0-9]+]].h, vl32
; VBITS_GE_2048-NEXT: ld1h { [[VALS:z[0-9]+]].h }, [[PG0]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG1:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: fcmeq [[CMP:p[0-9]+]].h, [[PG0]]/z, [[VALS]].h, #0.0
; VBITS_GE_2048-NEXT: mov [[MONE:z[0-9]+]].h, [[CMP]]/z, #-1
; VBITS_GE_2048-NEXT: uunpklo [[UPK1:z[0-9]+]].s, [[MONE]].h
; VBITS_GE_2048-NEXT: uunpklo [[UPKV1:z[0-9]+]].s, [[VALS]].h
; VBITS_GE_2048-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; VBITS_GE_2048-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG1]]/z, [[UPK2]].d, #0
; VBITS_GE_2048-NEXT: uunpklo [[UPKV2:z[0-9]+]].d, [[UPKV1]].s
; VBITS_GE_2048-NEXT: st1h { [[UPKV2]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; VBITS_GE_2048-NEXT: ret
  %vals = load <32 x half>, <32 x half>* %a
  %ptrs = load <32 x half*>, <32 x half*>* %b
  %mask = fcmp oeq <32 x half> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f16(<32 x half> %vals, <32 x half*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

;
; ST1W (float)
;

define void @masked_scatter_v2f32(<2 x float>* %a, <2 x float*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v2f32:
; CHECK: ldr d[[VALS:[0-9]+]], [x0]
; CHECK-NEXT: ldr q[[PTRS:[0-9]+]], [x1]
; CHECK-NEXT: ptrue [[PG0:p[0-9]+]].d, vl2
; CHECK-NEXT: fcmeq v[[CMP:[0-9]+]].2s, v[[VALS]].2s, #0
; CHECK-NEXT: ushll v[[SHLC:[0-9]+]].2d, v[[CMP]].2s, #0
; CHECK-NEXT: ushll v[[SHL:[0-9]+]].2d, v[[VALS]].2s, #0
; CHECK-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG0]]/z, z[[SHLC]].d, #0
; CHECK-NEXT: st1w { z[[SHL]].d }, [[MASK]], [z[[PTRS]].d]
; CHECK-NEXT: ret
  %vals = load <2 x float>, <2 x float>* %a
  %ptrs = load <2 x float*>, <2 x float*>* %b
  %mask = fcmp oeq <2 x float> %vals, zeroinitializer
  call void @llvm.masked.scatter.v2f32(<2 x float> %vals, <2 x float*> %ptrs, i32 8, <2 x i1> %mask)
  ret void
}

define void @masked_scatter_v4f32(<4 x float>* %a, <4 x float*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v4f32:
; CHECK: ldr q[[VALS:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-NEXT: fcmeq v[[CMP:[0-9]+]].4s, v[[VALS]].4s, #0
; CHECK-NEXT: uunpklo [[UPK:z[0-9]+]].d, z[[CMP]].s
; CHECK-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG]]/z, [[UPK]].d, #0
; CHECK-NEXT: uunpklo [[UPKV:z[0-9]+]].d, z[[VALS]].s
; CHECK-NEXT: st1w { [[UPKV]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; CHECK-NEXT: ret
  %vals = load <4 x float>, <4 x float>* %a
  %ptrs = load <4 x float*>, <4 x float*>* %b
  %mask = fcmp oeq <4 x float> %vals, zeroinitializer
  call void @llvm.masked.scatter.v4f32(<4 x float> %vals, <4 x float*> %ptrs, i32 8, <4 x i1> %mask)
  ret void
}

define void @masked_scatter_v8f32(<8 x float>* %a, <8 x float*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v8f32:
; VBITS_GE_512: ptrue [[PG0:p[0-9]+]].s, vl8
; VBITS_GE_512-NEXT: ld1w { [[VALS:z[0-9]+]].s }, [[PG0]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG1:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_512-NEXT: fcmeq [[CMP:p[0-9]+]].s, [[PG0]]/z, [[VALS]].s, #0.0
; VBITS_GE_512-NEXT: mov [[MONE:z[0-9]]].s, [[CMP]]/z, #-1
; VBITS_GE_512-NEXT: uunpklo [[UPK:z[0-9]+]].d, [[MONE]].s
; VBITS_GE_512-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG1]]/z, [[UPK]].d, #0
; VBITS_GE_512-NEXT: uunpklo [[UPKV:z[0-9]+]].d, [[VALS]].s
; VBITS_GE_512-NEXT: st1w { [[UPKV]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; VBITS_GE_512-NEXT: ret
  %vals = load <8 x float>, <8 x float>* %a
  %ptrs = load <8 x float*>, <8 x float*>* %b
  %mask = fcmp oeq <8 x float> %vals, zeroinitializer
  call void @llvm.masked.scatter.v8f32(<8 x float> %vals, <8 x float*> %ptrs, i32 8, <8 x i1> %mask)
  ret void
}

define void @masked_scatter_v16f32(<16 x float>* %a, <16 x float*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v16f32:
; VBITS_GE_1024: ptrue [[PG0:p[0-9]+]].s, vl16
; VBITS_GE_1024-NEXT: ld1w { [[VALS:z[0-9]+]].s }, [[PG0]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG1:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: fcmeq [[CMP:p[0-9]+]].s, [[PG0]]/z, [[VALS]].s, #0.0
; VBITS_GE_1024-NEXT: mov [[MONE:z[0-9]]].s, [[CMP]]/z, #-1
; VBITS_GE_1024-NEXT: uunpklo [[UPK:z[0-9]+]].d, [[MONE]].s
; VBITS_GE_1024-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG1]]/z, [[UPK]].d, #0
; VBITS_GE_1024-NEXT: uunpklo [[UPKV:z[0-9]+]].d, [[VALS]].s
; VBITS_GE_1024-NEXT: st1w { [[UPKV]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; VBITS_GE_1024-NEXT: ret
  %vals = load <16 x float>, <16 x float>* %a
  %ptrs = load <16 x float*>, <16 x float*>* %b
  %mask = fcmp oeq <16 x float> %vals, zeroinitializer
  call void @llvm.masked.scatter.v16f32(<16 x float> %vals, <16 x float*> %ptrs, i32 8, <16 x i1> %mask)
  ret void
}

define void @masked_scatter_v32f32(<32 x float>* %a, <32 x float*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v32f32:
; VBITS_GE_2048: ptrue [[PG0:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: ld1w { [[VALS:z[0-9]+]].s }, [[PG0]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG1:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: fcmeq [[CMP:p[0-9]+]].s, [[PG0]]/z, [[VALS]].s, #0.0
; VBITS_GE_2048-NEXT: mov [[MONE:z[0-9]]].s, [[CMP]]/z, #-1
; VBITS_GE_2048-NEXT: uunpklo [[UPK:z[0-9]+]].d, [[MONE]].s
; VBITS_GE_2048-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG1]]/z, [[UPK]].d, #0
; VBITS_GE_2048-NEXT: uunpklo [[UPKV:z[0-9]+]].d, [[VALS]].s
; VBITS_GE_2048-NEXT: st1w { [[UPKV]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; VBITS_GE_2048-NEXT: ret
  %vals = load <32 x float>, <32 x float>* %a
  %ptrs = load <32 x float*>, <32 x float*>* %b
  %mask = fcmp oeq <32 x float> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f32(<32 x float> %vals, <32 x float*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

;
; ST1D (float)
;

; Scalarize 1 x double scatters
define void @masked_scatter_v1f64(<1 x double>* %a, <1 x double*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v1f64:
; CHECK-NOT: ptrue
  %vals = load <1 x double>, <1 x double>* %a
  %ptrs = load <1 x double*>, <1 x double*>* %b
  %mask = fcmp oeq <1 x double> %vals, zeroinitializer
  call void @llvm.masked.scatter.v1f64(<1 x double> %vals, <1 x double*> %ptrs, i32 8, <1 x i1> %mask)
  ret void
}

define void @masked_scatter_v2f64(<2 x double>* %a, <2 x double*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v2f64:
; CHECK: ldr q[[VALS:[0-9]+]], [x0]
; CHECK-NEXT: ldr q[[PTRS:[0-9]+]], [x1]
; CHECK-NEXT: ptrue [[PG0:p[0-9]+]].d, vl2
; CHECK-NEXT: fcmeq v[[CMP:[0-9]+]].2d, v[[VALS]].2d, #0
; CHECK-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG0]]/z, z[[CMP]].d, #0
; CHECK-NEXT: st1d { z[[VALS]].d }, [[MASK]], [z[[PTRS]].d]
; CHECK-NEXT: ret
  %vals = load <2 x double>, <2 x double>* %a
  %ptrs = load <2 x double*>, <2 x double*>* %b
  %mask = fcmp oeq <2 x double> %vals, zeroinitializer
  call void @llvm.masked.scatter.v2f64(<2 x double> %vals, <2 x double*> %ptrs, i32 8, <2 x i1> %mask)
  ret void
}

define void @masked_scatter_v4f64(<4 x double>* %a, <4 x double*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v4f64:
; CHECK: ptrue [[PG0:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[VALS:z[0-9]+]].d }, [[PG0]]/z, [x0]
; CHECK-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG0]]/z, [x1]
; CHECK-NEXT: fcmeq [[MASK:p[0-9]+]].d, [[PG0]]/z, [[VALS]].d, #0.0
; CHECK-NEXT: st1d { [[VALS]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; CHECK-NEXT: ret
  %vals = load <4 x double>, <4 x double>* %a
  %ptrs = load <4 x double*>, <4 x double*>* %b
  %mask = fcmp oeq <4 x double> %vals, zeroinitializer
  call void @llvm.masked.scatter.v4f64(<4 x double> %vals, <4 x double*> %ptrs, i32 8, <4 x i1> %mask)
  ret void
}

define void @masked_scatter_v8f64(<8 x double>* %a, <8 x double*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v8f64:
; VBITS_GE_512: ptrue [[PG0:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[VALS:z[0-9]+]].d }, [[PG0]]/z, [x0]
; VBITS_GE_512-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG0]]/z, [x1]
; VBITS_GE_512-NEXT: fcmeq [[MASK:p[0-9]+]].d, [[PG0]]/z, [[VALS]].d, #0.0
; VBITS_GE_512-NEXT: st1d { [[VALS]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; VBITS_GE_512-NEXT: ret
  %vals = load <8 x double>, <8 x double>* %a
  %ptrs = load <8 x double*>, <8 x double*>* %b
  %mask = fcmp oeq <8 x double> %vals, zeroinitializer
  call void @llvm.masked.scatter.v8f64(<8 x double> %vals, <8 x double*> %ptrs, i32 8, <8 x i1> %mask)
  ret void
}

define void @masked_scatter_v16f64(<16 x double>* %a, <16 x double*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v16f64:
; VBITS_GE_1024: ptrue [[PG0:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[VALS:z[0-9]+]].d }, [[PG0]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG0]]/z, [x1]
; VBITS_GE_1024-NEXT: fcmeq [[MASK:p[0-9]+]].d, [[PG0]]/z, [[VALS]].d, #0.0
; VBITS_GE_1024-NEXT: st1d { [[VALS]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; VBITS_GE_1024-NEXT: ret
  %vals = load <16 x double>, <16 x double>* %a
  %ptrs = load <16 x double*>, <16 x double*>* %b
  %mask = fcmp oeq <16 x double> %vals, zeroinitializer
  call void @llvm.masked.scatter.v16f64(<16 x double> %vals, <16 x double*> %ptrs, i32 8, <16 x i1> %mask)
  ret void
}

define void @masked_scatter_v32f64(<32 x double>* %a, <32 x double*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v32f64:
; VBITS_GE_2048: ptrue [[PG0:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[VALS:z[0-9]+]].d }, [[PG0]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG0]]/z, [x1]
; VBITS_GE_2048-NEXT: fcmeq [[MASK:p[0-9]+]].d, [[PG0]]/z, [[VALS]].d, #0.0
; VBITS_GE_2048-NEXT: st1d { [[VALS]].d }, [[MASK]], {{\[}}[[PTRS]].d]
; VBITS_GE_2048-NEXT: ret
  %vals = load <32 x double>, <32 x double>* %a
  %ptrs = load <32 x double*>, <32 x double*>* %b
  %mask = fcmp oeq <32 x double> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f64(<32 x double> %vals, <32 x double*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

; The above tests test the types, the below tests check that the addressing
; modes still function
define void @masked_scatter_32b_scaled_sext_f16(<32 x half>* %a, <32 x i32>* %b, half* %base) #0 {
; CHECK-LABEL: masked_scatter_32b_scaled_sext_f16:
; VBITS_GE_2048: ptrue [[PG0:p[0-9]+]].h, vl32
; VBITS_GE_2048-NEXT: ld1h { [[VALS:z[0-9]+]].h }, [[PG0]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG1:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: ld1w { [[PTRS:z[0-9]+]].s }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: fcmeq [[CMP:p[0-9]+]].h, [[PG0]]/z, [[VALS]].h, #0.0
; VBITS_GE_2048-NEXT: mov [[MONE:z[0-9]+]].h, [[PG0]]/z, #-1
; VBITS_GE_2048-NEXT: uunpklo [[UPK:z[0-9]+]].s, [[MONE]].h
; VBITS_GE_2048-NEXT: cmpne [[MASK:p[0-9]+]].s, [[PG1]]/z, [[UPK]].s, #0
; VBITS_GE_2048-NEXT: uunpklo [[UPKV:z[0-9]+]].s, [[VALS]].h
; VBITS_GE_2048-NEXT: st1h { [[UPKV]].s }, [[MASK]], [x2, [[PTRS]].s, sxtw #1]
; VBITS_GE_2048-NEXT: ret
  %vals = load <32 x half>, <32 x half>* %a
  %idxs = load <32 x i32>, <32 x i32>* %b
  %ext = sext <32 x i32> %idxs to <32 x i64>
  %ptrs = getelementptr half, half* %base, <32 x i64> %ext
  %mask = fcmp oeq <32 x half> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f16(<32 x half> %vals, <32 x half*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

define void @masked_scatter_32b_scaled_sext_f32(<32 x float>* %a, <32 x i32>* %b, float* %base) #0 {
; CHECK-LABEL: masked_scatter_32b_scaled_sext_f32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: ld1w { [[VALS:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1w { [[PTRS:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: fcmeq [[MASK:p[0-9]+]].s, [[PG]]/z, [[VALS]].s, #0.0
; VBITS_GE_2048-NEXT: st1w { [[VALS]].s }, [[MASK]], [x2, [[PTRS]].s, sxtw #2]
; VBITS_GE_2048-NEXT: ret
  %vals = load <32 x float>, <32 x float>* %a
  %idxs = load <32 x i32>, <32 x i32>* %b
  %ext = sext <32 x i32> %idxs to <32 x i64>
  %ptrs = getelementptr float, float* %base, <32 x i64> %ext
  %mask = fcmp oeq <32 x float> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f32(<32 x float> %vals, <32 x float*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

define void @masked_scatter_32b_scaled_sext_f64(<32 x double>* %a, <32 x i32>* %b, double* %base) #0 {
; CHECK-LABEL: masked_scatter_32b_scaled_sext_f64:
; VBITS_GE_2048: ptrue [[PG0:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[VALS:z[0-9]+]].d }, [[PG0]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG1:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: ld1w { [[PTRS:z[0-9]+]].s }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: fcmeq [[MASK:p[0-9]+]].d, [[PG0]]/z, [[VALS]].d, #0.0
; VBITS_GE_2048-NEXT: st1d { [[VALS]].d }, [[MASK]], [x2, [[PTRS]].d, sxtw #3]
; VBITS_GE_2048-NEXT: ret
  %vals = load <32 x double>, <32 x double>* %a
  %idxs = load <32 x i32>, <32 x i32>* %b
  %ext = sext <32 x i32> %idxs to <32 x i64>
  %ptrs = getelementptr double, double* %base, <32 x i64> %ext
  %mask = fcmp oeq <32 x double> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f64(<32 x double> %vals, <32 x double*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

define void @masked_scatter_32b_scaled_zext(<32 x half>* %a, <32 x i32>* %b, half* %base) #0 {
; CHECK-LABEL: masked_scatter_32b_scaled_zext:
; VBITS_GE_2048: ptrue [[PG0:p[0-9]+]].h, vl32
; VBITS_GE_2048-NEXT: ld1h { [[VALS:z[0-9]+]].h }, [[PG0]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG1:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: ld1w { [[PTRS:z[0-9]+]].s }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: fcmeq [[CMP:p[0-9]+]].h, [[PG0]]/z, [[VALS]].h, #0.0
; VBITS_GE_2048-NEXT: mov [[MONE:z[0-9]+]].h, [[PG0]]/z, #-1
; VBITS_GE_2048-NEXT: uunpklo [[UPK:z[0-9]+]].s, [[MONE]].h
; VBITS_GE_2048-NEXT: cmpne [[MASK:p[0-9]+]].s, [[PG1]]/z, [[UPK]].s, #0
; VBITS_GE_2048-NEXT: uunpklo [[UPKV:z[0-9]+]].s, [[VALS]].h
; VBITS_GE_2048-NEXT: st1h { [[UPKV]].s }, [[MASK]], [x2, [[PTRS]].s, uxtw #1]
; VBITS_GE_2048-NEXT: ret
  %vals = load <32 x half>, <32 x half>* %a
  %idxs = load <32 x i32>, <32 x i32>* %b
  %ext = zext <32 x i32> %idxs to <32 x i64>
  %ptrs = getelementptr half, half* %base, <32 x i64> %ext
  %mask = fcmp oeq <32 x half> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f16(<32 x half> %vals, <32 x half*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

define void @masked_scatter_32b_unscaled_sext(<32 x half>* %a, <32 x i32>* %b, i8* %base) #0 {
; CHECK-LABEL: masked_scatter_32b_unscaled_sext:
; VBITS_GE_2048: ptrue [[PG0:p[0-9]+]].h, vl32
; VBITS_GE_2048-NEXT: ld1h { [[VALS:z[0-9]+]].h }, [[PG0]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG1:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: ld1w { [[PTRS:z[0-9]+]].s }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: fcmeq [[CMP:p[0-9]+]].h, [[PG0]]/z, [[VALS]].h, #0.0
; VBITS_GE_2048-NEXT: mov [[MONE:z[0-9]+]].h, [[PG0]]/z, #-1
; VBITS_GE_2048-NEXT: uunpklo [[UPK:z[0-9]+]].s, [[MONE]].h
; VBITS_GE_2048-NEXT: cmpne [[MASK:p[0-9]+]].s, [[PG1]]/z, [[UPK]].s, #0
; VBITS_GE_2048-NEXT: uunpklo [[UPKV:z[0-9]+]].s, [[VALS]].h
; VBITS_GE_2048-NEXT: st1h { [[UPKV]].s }, [[MASK]], [x2, [[PTRS]].s, sxtw]
; VBITS_GE_2048-NEXT: ret
  %vals = load <32 x half>, <32 x half>* %a
  %idxs = load <32 x i32>, <32 x i32>* %b
  %ext = sext <32 x i32> %idxs to <32 x i64>
  %byte_ptrs = getelementptr i8, i8* %base, <32 x i64> %ext
  %ptrs = bitcast <32 x i8*> %byte_ptrs to <32 x half*>
  %mask = fcmp oeq <32 x half> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f16(<32 x half> %vals, <32 x half*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

define void @masked_scatter_32b_unscaled_zext(<32 x half>* %a, <32 x i32>* %b, i8* %base) #0 {
; CHECK-LABEL: masked_scatter_32b_unscaled_zext:
; VBITS_GE_2048: ptrue [[PG0:p[0-9]+]].h, vl32
; VBITS_GE_2048-NEXT: ld1h { [[VALS:z[0-9]+]].h }, [[PG0]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG1:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: ld1w { [[PTRS:z[0-9]+]].s }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: fcmeq [[CMP:p[0-9]+]].h, [[PG0]]/z, [[VALS]].h, #0.0
; VBITS_GE_2048-NEXT: mov [[MONE:z[0-9]+]].h, [[PG0]]/z, #-1
; VBITS_GE_2048-NEXT: uunpklo [[UPK:z[0-9]+]].s, [[MONE]].h
; VBITS_GE_2048-NEXT: cmpne [[MASK:p[0-9]+]].s, [[PG1]]/z, [[UPK]].s, #0
; VBITS_GE_2048-NEXT: uunpklo [[UPKV:z[0-9]+]].s, [[VALS]].h
; VBITS_GE_2048-NEXT: st1h { [[UPKV]].s }, [[MASK]], [x2, [[PTRS]].s, uxtw]
; VBITS_GE_2048-NEXT: ret
  %vals = load <32 x half>, <32 x half>* %a
  %idxs = load <32 x i32>, <32 x i32>* %b
  %ext = zext <32 x i32> %idxs to <32 x i64>
  %byte_ptrs = getelementptr i8, i8* %base, <32 x i64> %ext
  %ptrs = bitcast <32 x i8*> %byte_ptrs to <32 x half*>
  %mask = fcmp oeq <32 x half> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f16(<32 x half> %vals, <32 x half*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

define void @masked_scatter_64b_scaled(<32 x float>* %a, <32 x i64>* %b, float* %base) #0 {
; CHECK-LABEL: masked_scatter_64b_scaled:
; VBITS_GE_2048: ptrue [[PG0:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: ld1w { [[VALS:z[0-9]+]].s }, [[PG0]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG1:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: fcmeq [[CMP:p[0-9]+]].s, [[PG0]]/z, [[VALS]].s, #0.0
; VBITS_GE_2048-NEXT: mov [[MONE:z[0-9]+]].s, [[PG0]]/z, #-1
; VBITS_GE_2048-NEXT: uunpklo [[UPK:z[0-9]+]].d, [[MONE]].s
; VBITS_GE_2048-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG1]]/z, [[UPK]].d, #0
; VBITS_GE_2048-NEXT: uunpklo [[UPKV:z[0-9]+]].d, [[VALS]].s
; VBITS_GE_2048-NEXT: st1w { [[UPKV]].d }, [[MASK]], [x2, [[PTRS]].d, lsl #2]
; VBITS_GE_2048-NEXT: ret
  %vals = load <32 x float>, <32 x float>* %a
  %idxs = load <32 x i64>, <32 x i64>* %b
  %ptrs = getelementptr float, float* %base, <32 x i64> %idxs
  %mask = fcmp oeq <32 x float> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f32(<32 x float> %vals, <32 x float*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

define void @masked_scatter_64b_unscaled(<32 x float>* %a, <32 x i64>* %b, i8* %base) #0 {
; CHECK-LABEL: masked_scatter_64b_unscaled:
; VBITS_GE_2048: ptrue [[PG0:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: ld1w { [[VALS:z[0-9]+]].s }, [[PG0]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG1:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: fcmeq [[CMP:p[0-9]+]].s, [[PG0]]/z, [[VALS]].s, #0.0
; VBITS_GE_2048-NEXT: mov [[MONE:z[0-9]+]].s, [[PG0]]/z, #-1
; VBITS_GE_2048-NEXT: uunpklo [[UPK:z[0-9]+]].d, [[MONE]].s
; VBITS_GE_2048-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG1]]/z, [[UPK]].d, #0
; VBITS_GE_2048-NEXT: uunpklo [[UPKV:z[0-9]+]].d, [[VALS]].s
; VBITS_GE_2048-NEXT: st1w { [[UPKV]].d }, [[MASK]], [x2, [[PTRS]].d]
; VBITS_GE_2048-NEXT: ret
  %vals = load <32 x float>, <32 x float>* %a
  %idxs = load <32 x i64>, <32 x i64>* %b
  %byte_ptrs = getelementptr i8, i8* %base, <32 x i64> %idxs
  %ptrs = bitcast <32 x i8*> %byte_ptrs to <32 x float*>
  %mask = fcmp oeq <32 x float> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f32(<32 x float> %vals, <32 x float*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

; FIXME: This case does not yet codegen well due to deficiencies in opcode selection
define void @masked_scatter_vec_plus_reg(<32 x float>* %a, <32 x i8*>* %b, i64 %off) #0 {
; CHECK-LABEL: masked_scatter_vec_plus_reg:
; VBITS_GE_2048: ptrue [[PG0:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: ptrue [[PG1:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1w { [[VALS:z[0-9]+]].s }, [[PG0]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: mov [[OFF:z[0-9]+]].d, x2
; VBITS_GE_2048-NEXT: fcmeq [[CMP:p[0-9]+]].s, [[PG0]]/z, [[VALS]].s, #0.0
; VBITS_GE_2048-NEXT: add [[PTRS_ADD:z[0-9]+]].d, [[PG1]]/m, [[PTRS]].d, [[OFF]].d
; VBITS_GE_2048-NEXT: mov [[MONE:z[0-9]+]].s, [[PG0]]/z, #-1
; VBITS_GE_2048-NEXT: uunpklo [[UPK:z[0-9]+]].d, [[MONE]].s
; VBITS_GE_2048-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG1]]/z, [[UPK]].d, #0
; VBITS_GE_2048-NEXT: uunpklo [[UPKV:z[0-9]+]].d, [[VALS]].s
; VBITS_GE_2048-NEXT: st1w { [[UPKV]].d }, [[MASK]], {{\[}}[[PTRS_ADD]].d]
; VBITS_GE_2048-NEXT: ret
  %vals = load <32 x float>, <32 x float>* %a
  %bases = load <32 x i8*>, <32 x i8*>* %b
  %byte_ptrs = getelementptr i8, <32 x i8*> %bases, i64 %off
  %ptrs = bitcast <32 x i8*> %byte_ptrs to <32 x float*>
  %mask = fcmp oeq <32 x float> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f32(<32 x float> %vals, <32 x float*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

; FIXME: This case does not yet codegen well due to deficiencies in opcode selection
define void @masked_scatter_vec_plus_imm(<32 x float>* %a, <32 x i8*>* %b) #0 {
; CHECK-LABEL: masked_scatter_vec_plus_imm:
; VBITS_GE_2048: ptrue [[PG0:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: ptrue [[PG1:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1w { [[VALS:z[0-9]+]].s }, [[PG0]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1d { [[PTRS:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: mov [[OFF:z[0-9]+]].d, #4
; VBITS_GE_2048-NEXT: fcmeq [[CMP:p[0-9]+]].s, [[PG0]]/z, [[VALS]].s, #0.0
; VBITS_GE_2048-NEXT: add [[PTRS_ADD:z[0-9]+]].d, [[PG1]]/m, [[PTRS]].d, [[OFF]].d
; VBITS_GE_2048-NEXT: mov [[MONE:z[0-9]+]].s, [[PG0]]/z, #-1
; VBITS_GE_2048-NEXT: uunpklo [[UPK:z[0-9]+]].d, [[MONE]].s
; VBITS_GE_2048-NEXT: cmpne [[MASK:p[0-9]+]].d, [[PG1]]/z, [[UPK]].d, #0
; VBITS_GE_2048-NEXT: uunpklo [[UPKV:z[0-9]+]].d, [[VALS]].s
; VBITS_GE_2048-NEXT: st1w { [[UPKV]].d }, [[MASK]], {{\[}}[[PTRS_ADD]].d]
; VBITS_GE_2048-NEXT: ret
  %vals = load <32 x float>, <32 x float>* %a
  %bases = load <32 x i8*>, <32 x i8*>* %b
  %byte_ptrs = getelementptr i8, <32 x i8*> %bases, i64 4
  %ptrs = bitcast <32 x i8*> %byte_ptrs to <32 x float*>
  %mask = fcmp oeq <32 x float> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f32(<32 x float> %vals, <32 x float*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

declare void @llvm.masked.scatter.v2i8(<2 x i8>, <2 x i8*>, i32, <2 x i1>)
declare void @llvm.masked.scatter.v4i8(<4 x i8>, <4 x i8*>, i32, <4 x i1>)
declare void @llvm.masked.scatter.v8i8(<8 x i8>, <8 x i8*>, i32, <8 x i1>)
declare void @llvm.masked.scatter.v16i8(<16 x i8>, <16 x i8*>, i32, <16 x i1>)
declare void @llvm.masked.scatter.v32i8(<32 x i8>, <32 x i8*>, i32, <32 x i1>)

declare void @llvm.masked.scatter.v2i16(<2 x i16>, <2 x i16*>, i32, <2 x i1>)
declare void @llvm.masked.scatter.v4i16(<4 x i16>, <4 x i16*>, i32, <4 x i1>)
declare void @llvm.masked.scatter.v8i16(<8 x i16>, <8 x i16*>, i32, <8 x i1>)
declare void @llvm.masked.scatter.v16i16(<16 x i16>, <16 x i16*>, i32, <16 x i1>)
declare void @llvm.masked.scatter.v32i16(<32 x i16>, <32 x i16*>, i32, <32 x i1>)

declare void @llvm.masked.scatter.v2i32(<2 x i32>, <2 x i32*>, i32, <2 x i1>)
declare void @llvm.masked.scatter.v4i32(<4 x i32>, <4 x i32*>, i32, <4 x i1>)
declare void @llvm.masked.scatter.v8i32(<8 x i32>, <8 x i32*>, i32, <8 x i1>)
declare void @llvm.masked.scatter.v16i32(<16 x i32>, <16 x i32*>, i32, <16 x i1>)
declare void @llvm.masked.scatter.v32i32(<32 x i32>, <32 x i32*>, i32, <32 x i1>)

declare void @llvm.masked.scatter.v1i64(<1 x i64>, <1 x i64*>, i32, <1 x i1>)
declare void @llvm.masked.scatter.v2i64(<2 x i64>, <2 x i64*>, i32, <2 x i1>)
declare void @llvm.masked.scatter.v4i64(<4 x i64>, <4 x i64*>, i32, <4 x i1>)
declare void @llvm.masked.scatter.v8i64(<8 x i64>, <8 x i64*>, i32, <8 x i1>)
declare void @llvm.masked.scatter.v16i64(<16 x i64>, <16 x i64*>, i32, <16 x i1>)
declare void @llvm.masked.scatter.v32i64(<32 x i64>, <32 x i64*>, i32, <32 x i1>)

declare void @llvm.masked.scatter.v2f16(<2 x half>, <2 x half*>, i32, <2 x i1>)
declare void @llvm.masked.scatter.v4f16(<4 x half>, <4 x half*>, i32, <4 x i1>)
declare void @llvm.masked.scatter.v8f16(<8 x half>, <8 x half*>, i32, <8 x i1>)
declare void @llvm.masked.scatter.v16f16(<16 x half>, <16 x half*>, i32, <16 x i1>)
declare void @llvm.masked.scatter.v32f16(<32 x half>, <32 x half*>, i32, <32 x i1>)

declare void @llvm.masked.scatter.v2f32(<2 x float>, <2 x float*>, i32, <2 x i1>)
declare void @llvm.masked.scatter.v4f32(<4 x float>, <4 x float*>, i32, <4 x i1>)
declare void @llvm.masked.scatter.v8f32(<8 x float>, <8 x float*>, i32, <8 x i1>)
declare void @llvm.masked.scatter.v16f32(<16 x float>, <16 x float*>, i32, <16 x i1>)
declare void @llvm.masked.scatter.v32f32(<32 x float>, <32 x float*>, i32, <32 x i1>)

declare void @llvm.masked.scatter.v1f64(<1 x double>, <1 x double*>, i32, <1 x i1>)
declare void @llvm.masked.scatter.v2f64(<2 x double>, <2 x double*>, i32, <2 x i1>)
declare void @llvm.masked.scatter.v4f64(<4 x double>, <4 x double*>, i32, <4 x i1>)
declare void @llvm.masked.scatter.v8f64(<8 x double>, <8 x double*>, i32, <8 x i1>)
declare void @llvm.masked.scatter.v16f64(<16 x double>, <16 x double*>, i32, <16 x i1>)
declare void @llvm.masked.scatter.v32f64(<32 x double>, <32 x double*>, i32, <32 x i1>)

attributes #0 = { "target-features"="+sve" }
