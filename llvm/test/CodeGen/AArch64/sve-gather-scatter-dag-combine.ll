; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

; Verify that DAG combine rules for LD1 + sext/zext don't apply when the
; result of LD1 has multiple uses

define <vscale x 2 x i64> @no_dag_combine_zext_sext(<vscale x 2 x i1> %pg,
                                                    <vscale x 2 x i64> %base,
                                                    <vscale x 2 x i8>* %res_out,
                                                    <vscale x 2 x i1> %pred) {
; CHECK-LABEL: no_dag_combine_zext_sext
; CHECK:  	ld1b	{ z0.d }, p0/z, [z0.d, #16]
; CHECK-NEXT:	st1b	{ z0.d }, p1, [x0]
; CHECK-NEXT:	and	z0.d, z0.d, #0xff
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.scalar.offset.nxv2i8.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                           <vscale x 2 x i64> %base,
                                                                                           i64 16)
  %res1 = zext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  %res2 = sext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  call void @llvm.masked.store.nxv2i8(<vscale x 2 x i8> %load,
                                      <vscale x 2 x i8> *%res_out,
                                      i32 8,
                                      <vscale x 2 x i1> %pred)

  ret <vscale x 2 x i64> %res1
}

define <vscale x 2 x i64> @no_dag_combine_sext(<vscale x 2 x i1> %pg,
                                               <vscale x 2 x i64> %base,
                                               <vscale x 2 x i8>* %res_out,
                                               <vscale x 2 x i1> %pred) {
; CHECK-LABEL: no_dag_combine_sext
; CHECK:  	ld1b	{ z1.d }, p0/z, [z0.d, #16]
; CHECK-NEXT:	ptrue	p0.d
; CHECK-NEXT: movprfx z0, z1
; CHECK-NEXT:	sxtb	z0.d, p0/m, z1.d
; CHECK-NEXT:	st1b	{ z1.d }, p1, [x0]
; CHECK-NEXT:	ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.scalar.offset.nxv2i8.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                           <vscale x 2 x i64> %base,
                                                                                           i64 16)
  %res = sext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  call void @llvm.masked.store.nxv2i8(<vscale x 2 x i8> %load,
                                      <vscale x 2 x i8> *%res_out,
                                      i32 8,
                                      <vscale x 2 x i1> %pred)

  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @no_dag_combine_zext(<vscale x 2 x i1> %pg,
                                               <vscale x 2 x i64> %base,
                                               <vscale x 2 x i8>* %res_out,
                                               <vscale x 2 x i1> %pred) {
; CHECK-LABEL: no_dag_combine_zext
; CHECK:  	ld1b	{ z0.d }, p0/z, [z0.d, #16]
; CHECK-NEXT:	st1b	{ z0.d }, p1, [x0]
; CHECK-NEXT:	and	z0.d, z0.d, #0xff
; CHECK-NEXT:	ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.scalar.offset.nxv2i8.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                           <vscale x 2 x i64> %base,
                                                                                           i64 16)
  %res = zext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  call void @llvm.masked.store.nxv2i8(<vscale x 2 x i8> %load,
                                      <vscale x 2 x i8> *%res_out,
                                      i32 8,
                                      <vscale x 2 x i1> %pred)

  ret <vscale x 2 x i64> %res
}

define <vscale x 16 x i8> @narrow_i64_gather_index_i8_zext(i8* %out, i8* %in, <vscale x 16 x i8> %d, i64 %ptr){
; CHECK-LABEL: narrow_i64_gather_index_i8_zext:
; CHECK:       // %bb.0:
; CHECK-NEXT:    add x8, x1, x2
; CHECK-NEXT:    ptrue p0.s
; CHECK-NEXT:    ld1b { z0.s }, p0/z, [x1, x2]
; CHECK-NEXT:    ld1b { z1.s }, p0/z, [x8, #1, mul vl]
; CHECK-NEXT:    ld1b { z2.s }, p0/z, [x8, #2, mul vl]
; CHECK-NEXT:    ld1b { z3.s }, p0/z, [x8, #3, mul vl]
; CHECK-NEXT:    ld1b { z3.s }, p0/z, [x1, z3.s, uxtw]
; CHECK-NEXT:    ld1b { z2.s }, p0/z, [x1, z2.s, uxtw]
; CHECK-NEXT:    ld1b { z0.s }, p0/z, [x1, z0.s, uxtw]
; CHECK-NEXT:    ld1b { z1.s }, p0/z, [x1, z1.s, uxtw]
; CHECK-NEXT:    uzp1 z2.h, z2.h, z3.h
; CHECK-NEXT:    uzp1 z0.h, z0.h, z1.h
; CHECK-NEXT:    uzp1 z0.b, z0.b, z2.b
; CHECK-NEXT:    ret
  %1 = getelementptr inbounds i8, i8* %in, i64 %ptr
  %2 = bitcast i8* %1 to <vscale x 16 x i8>*
  %wide.load = load <vscale x 16 x i8>, <vscale x 16 x i8>* %2, align 1
  %3 = zext <vscale x 16 x i8> %wide.load to <vscale x 16 x i64>
  %4 = getelementptr inbounds i8, i8* %in, <vscale x 16 x i64> %3
  %wide.masked.gather = call <vscale x 16 x i8> @llvm.masked.gather.nxv16i8.nxv16p0(<vscale x 16 x i8*> %4, i32 1, <vscale x 16 x i1> shufflevector (<vscale x 16 x i1> insertelement (<vscale x 16 x i1> poison, i1 true, i32 0), <vscale x 16 x i1> poison, <vscale x 16 x i32> zeroinitializer), <vscale x 16 x i8> undef)
  ret <vscale x 16 x i8> %wide.masked.gather
}

define <vscale x 16 x i8> @narrow_i64_gather_index_i8_sext(i8* %out, i8* %in, <vscale x 16 x i8> %d, i64 %ptr){
; CHECK-LABEL: narrow_i64_gather_index_i8_sext:
; CHECK:       // %bb.0:
; CHECK-NEXT:    add x8, x1, x2
; CHECK-NEXT:    ptrue p0.s
; CHECK-NEXT:    ld1sb { z0.s }, p0/z, [x1, x2]
; CHECK-NEXT:    ld1sb { z1.s }, p0/z, [x8, #1, mul vl]
; CHECK-NEXT:    ld1sb { z2.s }, p0/z, [x8, #2, mul vl]
; CHECK-NEXT:    ld1sb { z3.s }, p0/z, [x8, #3, mul vl]
; CHECK-NEXT:    ld1b { z3.s }, p0/z, [x1, z3.s, sxtw]
; CHECK-NEXT:    ld1b { z2.s }, p0/z, [x1, z2.s, sxtw]
; CHECK-NEXT:    ld1b { z0.s }, p0/z, [x1, z0.s, sxtw]
; CHECK-NEXT:    ld1b { z1.s }, p0/z, [x1, z1.s, sxtw]
; CHECK-NEXT:    uzp1 z2.h, z2.h, z3.h
; CHECK-NEXT:    uzp1 z0.h, z0.h, z1.h
; CHECK-NEXT:    uzp1 z0.b, z0.b, z2.b
; CHECK-NEXT:    ret
  %1 = getelementptr inbounds i8, i8* %in, i64 %ptr
  %2 = bitcast i8* %1 to <vscale x 16 x i8>*
  %wide.load = load <vscale x 16 x i8>, <vscale x 16 x i8>* %2, align 1
  %3 = sext <vscale x 16 x i8> %wide.load to <vscale x 16 x i64>
  %4 = getelementptr inbounds i8, i8* %in, <vscale x 16 x i64> %3
  %wide.masked.gather = call <vscale x 16 x i8> @llvm.masked.gather.nxv16i8.nxv16p0(<vscale x 16 x i8*> %4, i32 1, <vscale x 16 x i1> shufflevector (<vscale x 16 x i1> insertelement (<vscale x 16 x i1> poison, i1 true, i32 0), <vscale x 16 x i1> poison, <vscale x 16 x i32> zeroinitializer), <vscale x 16 x i8> undef)
  ret <vscale x 16 x i8> %wide.masked.gather
}

define <vscale x 8 x i16> @narrow_i64_gather_index_i16_zext(i16* %out, i16* %in, <vscale x 8 x i16> %d, i64 %ptr){
; CHECK-LABEL: narrow_i64_gather_index_i16_zext:
; CHECK:       // %bb.0:
; CHECK-NEXT:    add x8, x1, x2, lsl #1
; CHECK-NEXT:    ptrue p0.s
; CHECK-NEXT:    ld1h { z0.s }, p0/z, [x1, x2, lsl #1]
; CHECK-NEXT:    ld1h { z1.s }, p0/z, [x8, #1, mul vl]
; CHECK-NEXT:    ld1h { z0.s }, p0/z, [x1, z0.s, uxtw #1]
; CHECK-NEXT:    ld1h { z1.s }, p0/z, [x1, z1.s, uxtw #1]
; CHECK-NEXT:    uzp1 z0.h, z0.h, z1.h
; CHECK-NEXT:    ret
  %1 = getelementptr inbounds i16, i16* %in, i64 %ptr
  %2 = bitcast i16* %1 to <vscale x 8 x i16>*
  %wide.load = load <vscale x 8 x i16>, <vscale x 8 x i16>* %2, align 1
  %3 = zext <vscale x 8 x i16> %wide.load to <vscale x 8 x i64>
  %4 = getelementptr inbounds i16, i16* %in, <vscale x 8 x i64> %3
  %wide.masked.gather = call <vscale x 8 x i16> @llvm.masked.gather.nxv8i16.nxv8p0(<vscale x 8 x i16*> %4, i32 1, <vscale x 8 x i1> shufflevector (<vscale x 8 x i1> insertelement (<vscale x 8 x i1> poison, i1 true, i32 0), <vscale x 8 x i1> poison, <vscale x 8 x i32> zeroinitializer), <vscale x 8 x i16> undef)
  ret <vscale x 8 x i16> %wide.masked.gather
}

define <vscale x 8 x i16> @narrow_i64_gather_index_i16_sext(i16* %out, i16* %in, <vscale x 8 x i16> %d, i64 %ptr){
; CHECK-LABEL: narrow_i64_gather_index_i16_sext:
; CHECK:       // %bb.0:
; CHECK-NEXT:    add x8, x1, x2, lsl #1
; CHECK-NEXT:    ptrue p0.s
; CHECK-NEXT:    ld1sh { z0.s }, p0/z, [x1, x2, lsl #1]
; CHECK-NEXT:    ld1sh { z1.s }, p0/z, [x8, #1, mul vl]
; CHECK-NEXT:    ld1h { z0.s }, p0/z, [x1, z0.s, sxtw #1]
; CHECK-NEXT:    ld1h { z1.s }, p0/z, [x1, z1.s, sxtw #1]
; CHECK-NEXT:    uzp1 z0.h, z0.h, z1.h
; CHECK-NEXT:    ret
  %1 = getelementptr inbounds i16, i16* %in, i64 %ptr
  %2 = bitcast i16* %1 to <vscale x 8 x i16>*
  %wide.load = load <vscale x 8 x i16>, <vscale x 8 x i16>* %2, align 1
  %3 = sext <vscale x 8 x i16> %wide.load to <vscale x 8 x i64>
  %4 = getelementptr inbounds i16, i16* %in, <vscale x 8 x i64> %3
  %wide.masked.gather = call <vscale x 8 x i16> @llvm.masked.gather.nxv8i16.nxv8p0(<vscale x 8 x i16*> %4, i32 1, <vscale x 8 x i1> shufflevector (<vscale x 8 x i1> insertelement (<vscale x 8 x i1> poison, i1 true, i32 0), <vscale x 8 x i1> poison, <vscale x 8 x i32> zeroinitializer), <vscale x 8 x i16> undef)
  ret <vscale x 8 x i16> %wide.masked.gather
}

define <vscale x 4 x i32> @no_narrow_i64_gather_index_i32(i32* %out, i32* %in, <vscale x 4 x i32> %d, i64 %ptr){
; CHECK-LABEL: no_narrow_i64_gather_index_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.s
; CHECK-NEXT:    ld1w { z0.s }, p0/z, [x1, x2, lsl #2]
; CHECK-NEXT:    ld1w { z0.s }, p0/z, [x1, z0.s, uxtw #2]
; CHECK-NEXT:    ret
  %1 = getelementptr inbounds i32, i32* %in, i64 %ptr
  %2 = bitcast i32* %1 to <vscale x 4 x i32>*
  %wide.load = load <vscale x 4 x i32>, <vscale x 4 x i32>* %2, align 1
  %3 = zext <vscale x 4 x i32> %wide.load to <vscale x 4 x i64>
  %4 = getelementptr inbounds i32, i32* %in, <vscale x 4 x i64> %3
  %wide.masked.gather = call <vscale x 4 x i32> @llvm.masked.gather.nxv4i32.nxv4p0(<vscale x 4 x i32*> %4, i32 1, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i32 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i32> undef)
  ret <vscale x 4 x i32> %wide.masked.gather
}

define <vscale x 2 x i64> @no_narrow_i64_gather_index_i64(i64* %out, i64* %in, <vscale x 2 x i64> %d, i64 %ptr){
; CHECK-LABEL: no_narrow_i64_gather_index_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x1, x2, lsl #3]
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x1, z0.d, lsl #3]
; CHECK-NEXT:    ret
  %1 = getelementptr inbounds i64, i64* %in, i64 %ptr
  %2 = bitcast i64* %1 to <vscale x 2 x i64>*
  %wide.load = load <vscale x 2 x i64>, <vscale x 2 x i64>* %2, align 1
  %3 = getelementptr inbounds i64, i64* %in, <vscale x 2 x i64> %wide.load
  %wide.masked.gather = call <vscale x 2 x i64> @llvm.masked.gather.nxv2i64.nxv2p0(<vscale x 2 x i64*> %3, i32 1, <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), <vscale x 2 x i64> undef)
  ret <vscale x 2 x i64> %wide.masked.gather
}

declare <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.scalar.offset.nxv2i8.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, i64)
declare void @llvm.masked.store.nxv2i8(<vscale x 2 x i8>, <vscale x 2 x i8>*, i32, <vscale x 2 x i1>)
declare <vscale x 16 x i8> @llvm.masked.gather.nxv16i8.nxv16p0(<vscale x 16 x i8*>, i32, <vscale x 16 x i1>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.masked.gather.nxv8i16.nxv8p0(<vscale x 8 x i16*>, i32, <vscale x 8 x i1>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.masked.gather.nxv4i32.nxv4p0(<vscale x 4 x i32*>, i32, <vscale x 4 x i1>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.masked.gather.nxv2i64.nxv2p0(<vscale x 2 x i64*>, i32, <vscale x 2 x i1>, <vscale x 2 x i64>)
