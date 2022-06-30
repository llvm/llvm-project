; RUN: opt < %s -interleaved-access -S | FileCheck %s

target triple = "aarch64-linux-gnu"

define void @load_factor2(<32 x i16>* %ptr) #0 {
; CHECK-LABEL:    @load_factor2(
; CHECK-NEXT:       [[PTRUE:%.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
; CHECK-NEXT:       [[TMP1:%.*]] = bitcast <32 x i16>* %ptr to i16*
; CHECK-NEXT:       [[LDN:%.*]] = call { <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sve.ld2.sret.nxv8i16(<vscale x 8 x i1> [[PTRUE]], i16* [[TMP1]])
; CHECK-NEXT:       [[TMP2:%.*]] = extractvalue { <vscale x 8 x i16>, <vscale x 8 x i16> } [[LDN]], 1
; CHECK-NEXT:       [[EXT1:%.*]] = call <16 x i16> @llvm.vector.extract.v16i16.nxv8i16(<vscale x 8 x i16> [[TMP2]], i64 0)
; CHECK-NEXT:       [[TMP3:%.*]] = extractvalue { <vscale x 8 x i16>, <vscale x 8 x i16> } [[LDN]], 0
; CHECK-NEXT:       [[EXT2:%.*]] = call <16 x i16> @llvm.vector.extract.v16i16.nxv8i16(<vscale x 8 x i16> [[TMP3]], i64 0)
; CHECK-NEXT:       ret void
  %interleaved.vec = load <32 x i16>, <32 x i16>* %ptr, align 4
  %v0 = shufflevector <32 x i16> %interleaved.vec, <32 x i16> poison, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14,
                                                                                  i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %v1 = shufflevector <32 x i16> %interleaved.vec, <32 x i16> poison, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15,
                                                                                  i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  ret void
}

define void @load_factor3(<24 x i32>* %ptr) #0 {
; CHECK-LABEL:    @load_factor3(
; CHECK-NEXT:       [[PTRUE:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
; CHECK-NEXT:       [[TMP1:%.*]] = bitcast <24 x i32>* %ptr to i32*
; CHECK-NEXT:       [[LDN:%.*]] = call { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.aarch64.sve.ld3.sret.nxv4i32(<vscale x 4 x i1> [[PTRUE]], i32* [[TMP1]])
; CHECK-NEXT:       [[TMP2:%.*]] = extractvalue { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> } [[LDN]], 2
; CHECK-NEXT:       [[EXT1:%.*]] = call <8 x i32> @llvm.vector.extract.v8i32.nxv4i32(<vscale x 4 x i32> [[TMP2]], i64 0)
; CHECK-NEXT:       [[TMP3:%.*]] = extractvalue { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> } [[LDN]], 1
; CHECK-NEXT:       [[EXT2:%.*]] = call <8 x i32> @llvm.vector.extract.v8i32.nxv4i32(<vscale x 4 x i32> [[TMP3]], i64 0)
; CHECK-NEXT:       [[TMP4:%.*]] = extractvalue { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> } [[LDN]], 0
; CHECK-NEXT:       [[EXT3:%.*]] = call <8 x i32> @llvm.vector.extract.v8i32.nxv4i32(<vscale x 4 x i32> [[TMP4]], i64 0)
; CHECK-NEXT:       ret void
  %interleaved.vec = load <24 x i32>, <24 x i32>* %ptr, align 4
  %v0 = shufflevector <24 x i32> %interleaved.vec, <24 x i32> poison, <8 x i32> <i32 0, i32 3, i32 6, i32 9, i32 12, i32 15, i32 18, i32 21>
  %v1 = shufflevector <24 x i32> %interleaved.vec, <24 x i32> poison, <8 x i32> <i32 1, i32 4, i32 7, i32 10, i32 13, i32 16, i32 19, i32 22>
  %v2 = shufflevector <24 x i32> %interleaved.vec, <24 x i32> poison, <8 x i32> <i32 2, i32 5, i32 8, i32 11, i32 14, i32 17, i32 20, i32 23>
  ret void
}

define void @load_factor4(<16 x i64>* %ptr) #0 {
; CHECK-LABEL:    @load_factor4(
; CHECK-NEXT:       [[PTRUE:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
; CHECK-NEXT:       [[TMP1:%.*]] = bitcast <16 x i64>* %ptr to i64*
; CHECK-NEXT:       [[LDN:%.*]] = call { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.ld4.sret.nxv2i64(<vscale x 2 x i1> [[PTRUE]], i64* [[TMP1]])
; CHECK-NEXT:       [[TMP2:%.*]] = extractvalue { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } [[LDN]], 3
; CHECK-NEXT:       [[EXT1:%.*]] = call <4 x i64> @llvm.vector.extract.v4i64.nxv2i64(<vscale x 2 x i64> [[TMP2]], i64 0)
; CHECK-NEXT:       [[TMP3:%.*]] = extractvalue { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } [[LDN]], 2
; CHECK-NEXT:       [[EXT2:%.*]] = call <4 x i64> @llvm.vector.extract.v4i64.nxv2i64(<vscale x 2 x i64> [[TMP3]], i64 0)
; CHECK-NEXT:       [[TMP4:%.*]] = extractvalue { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } [[LDN]], 1
; CHECK-NEXT:       [[EXT3:%.*]] = call <4 x i64> @llvm.vector.extract.v4i64.nxv2i64(<vscale x 2 x i64> [[TMP4]], i64 0)
; CHECK-NEXT:       [[TMP5:%.*]] = extractvalue { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } [[LDN]], 0
; CHECK-NEXT:       [[EXT4:%.*]] = call <4 x i64> @llvm.vector.extract.v4i64.nxv2i64(<vscale x 2 x i64> [[TMP5]], i64 0)
; CHECK-NEXT:       ret void
  %interleaved.vec = load <16 x i64>, <16 x i64>* %ptr, align 4
  %v0 = shufflevector <16 x i64> %interleaved.vec, <16 x i64> poison, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %v1 = shufflevector <16 x i64> %interleaved.vec, <16 x i64> poison, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %v2 = shufflevector <16 x i64> %interleaved.vec, <16 x i64> poison, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %v3 = shufflevector <16 x i64> %interleaved.vec, <16 x i64> poison, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  ret void
}

define void @store_factor2(<32 x i16>* %ptr, <16 x i16> %v0, <16 x i16> %v1) #0 {
; CHECK-LABEL:    @store_factor2(
; CHECK-NEXT:       [[PTRUE:%.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
; CHECK-NEXT:       [[TMP1:%.*]] = shufflevector <16 x i16> %v0, <16 x i16> %v1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
; CHECK-NEXT:       [[INS1:%.*]] = call <vscale x 8 x i16> @llvm.vector.insert.nxv8i16.v16i16(<vscale x 8 x i16> undef, <16 x i16> [[TMP1]], i64 0)
; CHECK-NEXT:       [[TMP2:%.*]] = shufflevector <16 x i16> %v0, <16 x i16> %v1, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
; CHECK-NEXT:       [[INS2:%.*]] = call <vscale x 8 x i16> @llvm.vector.insert.nxv8i16.v16i16(<vscale x 8 x i16> undef, <16 x i16> [[TMP2]], i64 0)
; CHECK-NEXT:       [[PTR:%.*]] = bitcast <32 x i16>* %ptr to i16*
; CHECK-NEXT:       call void @llvm.aarch64.sve.st2.nxv8i16(<vscale x 8 x i16> [[INS1]], <vscale x 8 x i16> [[INS2]], <vscale x 8 x i1> [[PTRUE]], i16* [[PTR]])
; CHECK-NEXT:       ret void
  %interleaved.vec = shufflevector <16 x i16> %v0, <16 x i16> %v1, <32 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23,
                                                                               i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  store <32 x i16> %interleaved.vec, <32 x i16>* %ptr, align 4
  ret void
}

define void @store_factor3(<24 x i32>* %ptr, <8 x i32> %v0, <8 x i32> %v1, <8 x i32> %v2) #0 {
; CHECK-LABEL:    @store_factor3(
; CHECK:            [[PTRUE:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
; CHECK-NEXT:       [[TMP1:%.*]] = shufflevector <16 x i32> %s0, <16 x i32> %s1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK-NEXT:       [[INS1:%.*]] = call <vscale x 4 x i32> @llvm.vector.insert.nxv4i32.v8i32(<vscale x 4 x i32> undef, <8 x i32> [[TMP1]], i64 0)
; CHECK-NEXT:       [[TMP2:%.*]] = shufflevector <16 x i32> %s0, <16 x i32> %s1, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
; CHECK-NEXT:       [[INS2:%.*]] = call <vscale x 4 x i32> @llvm.vector.insert.nxv4i32.v8i32(<vscale x 4 x i32> undef, <8 x i32> [[TMP2]], i64 0)
; CHECK-NEXT:       [[TMP3:%.*]] = shufflevector <16 x i32> %s0, <16 x i32> %s1, <8 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
; CHECK-NEXT:       [[INS3:%.*]] = call <vscale x 4 x i32> @llvm.vector.insert.nxv4i32.v8i32(<vscale x 4 x i32> undef, <8 x i32> [[TMP3]], i64 0)
; CHECK-NEXT:       [[PTR:%.*]] = bitcast <24 x i32>* %ptr to i32*
; CHECK-NEXT:       call void @llvm.aarch64.sve.st3.nxv4i32(<vscale x 4 x i32> [[INS1]], <vscale x 4 x i32> [[INS2]], <vscale x 4 x i32> [[INS3]], <vscale x 4 x i1> [[PTRUE]], i32* [[PTR]])
; CHECK-NEXT:       ret void
  %s0 = shufflevector <8 x i32> %v0, <8 x i32> %v1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %s1 = shufflevector <8 x i32> %v2, <8 x i32> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                   i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %interleaved.vec = shufflevector <16 x i32> %s0, <16 x i32> %s1, <24 x i32> <i32 0, i32 8, i32 16, i32 1, i32 9, i32 17, i32 2, i32 10, i32 18, i32 3, i32 11, i32 19,
                                                                               i32 4, i32 12, i32 20, i32 5, i32 13, i32 21, i32 6, i32 14, i32 22, i32 7, i32 15, i32 23>
  store <24 x i32> %interleaved.vec, <24 x i32>* %ptr, align 4
  ret void
}

define void @store_factor4(<16 x i64>* %ptr, <4 x i64> %v0, <4 x i64> %v1, <4 x i64> %v2, <4 x i64> %v3) #0 {
; CHECK-LABEL:    @store_factor4(
; CHECK:            [[PTRUE:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
; CHECK-NEXT:       [[TMP1:%.*]] = shufflevector <8 x i64> %s0, <8 x i64> %s1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT:       [[INS1:%.*]] = call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v4i64(<vscale x 2 x i64> undef, <4 x i64> [[TMP1]], i64 0)
; CHECK-NEXT:       [[TMP2:%.*]] = shufflevector <8 x i64> %s0, <8 x i64> %s1, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; CHECK-NEXT:       [[INS2:%.*]] = call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v4i64(<vscale x 2 x i64> undef, <4 x i64> [[TMP2]], i64 0)
; CHECK-NEXT:       [[TMP3:%.*]] = shufflevector <8 x i64> %s0, <8 x i64> %s1, <4 x i32> <i32 8, i32 9, i32 10, i32 11>
; CHECK-NEXT:       [[INS3:%.*]] = call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v4i64(<vscale x 2 x i64> undef, <4 x i64> [[TMP3]], i64 0)
; CHECK-NEXT:       [[TMP4:%.*]] = shufflevector <8 x i64> %s0, <8 x i64> %s1, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
; CHECK-NEXT:       [[INS4:%.*]] = call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v4i64(<vscale x 2 x i64> undef, <4 x i64> [[TMP4]], i64 0)
; CHECK-NEXT:       [[PTR:%.*]] = bitcast <16 x i64>* %ptr to i64*
; CHECK-NEXT:       call void @llvm.aarch64.sve.st4.nxv2i64(<vscale x 2 x i64> [[INS1]], <vscale x 2 x i64> [[INS2]], <vscale x 2 x i64> [[INS3]], <vscale x 2 x i64> [[INS4]], <vscale x 2 x i1> [[PTRUE]], i64* [[PTR]])
; CHECK-NEXT:       ret void
  %s0 = shufflevector <4 x i64> %v0, <4 x i64> %v1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %s1 = shufflevector <4 x i64> %v2, <4 x i64> %v3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %interleaved.vec = shufflevector <8 x i64> %s0, <8 x i64> %s1, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  store <16 x i64> %interleaved.vec, <16 x i64>* %ptr, align 4
  ret void
}

define void @load_ptrvec_factor2(<8 x i32*>* %ptr) #0 {
; CHECK-LABEL:    @load_ptrvec_factor2(
; CHECK-NEXT:       [[PTRUE:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
; CHECK-NEXT:       [[TMP1:%.*]] = bitcast <8 x i32*>* %ptr to i64*
; CHECK-NEXT:       [[LDN:%.*]] = call { <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.ld2.sret.nxv2i64(<vscale x 2 x i1> [[PTRUE]], i64* [[TMP1]])
; CHECK-NEXT:       [[TMP2:%.*]] = extractvalue { <vscale x 2 x i64>, <vscale x 2 x i64> } [[LDN]], 1
; CHECK-NEXT:       [[EXT1:%.*]] = call <4 x i64> @llvm.vector.extract.v4i64.nxv2i64(<vscale x 2 x i64> [[TMP2]], i64 0)
; CHECK-NEXT:       [[TOP1:%.*]] = inttoptr <4 x i64> [[EXT1]] to <4 x i32*>
; CHECK-NEXT:       [[TMP3:%.*]] = extractvalue { <vscale x 2 x i64>, <vscale x 2 x i64> } [[LDN]], 0
; CHECK-NEXT:       [[EXT2:%.*]] = call <4 x i64> @llvm.vector.extract.v4i64.nxv2i64(<vscale x 2 x i64> [[TMP3]], i64 0)
; CHECK-NEXT:       [[TOP2:%.*]] = inttoptr <4 x i64> [[EXT2]] to <4 x i32*>
; CHECK-NEXT:       ret void
  %interleaved.vec = load <8 x i32*>, <8 x i32*>* %ptr, align 4
  %v0 = shufflevector <8 x i32*> %interleaved.vec, <8 x i32*> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %v1 = shufflevector <8 x i32*> %interleaved.vec, <8 x i32*> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  ret void
}

define void @load_ptrvec_factor3(<12 x i32*>* %ptr) #0 {
; CHECK-LABEL:    @load_ptrvec_factor3(
; CHECK-NEXT:       [[PTRUE:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
; CHECK-NEXT:       [[TMP1:%.*]] = bitcast <12 x i32*>* %ptr to i64*
; CHECK-NEXT:       [[LDN:%.*]] = call { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.ld3.sret.nxv2i64(<vscale x 2 x i1> [[PTRUE]], i64* [[TMP1]])
; CHECK-NEXT:       [[TMP2:%.*]] = extractvalue { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } [[LDN]], 2
; CHECK-NEXT:       [[EXT1:%.*]] = call <4 x i64> @llvm.vector.extract.v4i64.nxv2i64(<vscale x 2 x i64> [[TMP2]], i64 0)
; CHECK-NEXT:       [[TOP1:%.*]] = inttoptr <4 x i64> [[EXT1]] to <4 x i32*>
; CHECK-NEXT:       [[TMP3:%.*]] = extractvalue { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } [[LDN]], 1
; CHECK-NEXT:       [[EXT2:%.*]] = call <4 x i64> @llvm.vector.extract.v4i64.nxv2i64(<vscale x 2 x i64> [[TMP3]], i64 0)
; CHECK-NEXT:       [[TOP2:%.*]] = inttoptr <4 x i64> [[EXT2]] to <4 x i32*>
; CHECK-NEXT:       [[TMP4:%.*]] = extractvalue { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } [[LDN]], 0
; CHECK-NEXT:       [[EXT3:%.*]] = call <4 x i64> @llvm.vector.extract.v4i64.nxv2i64(<vscale x 2 x i64> [[TMP4]], i64 0)
; CHECK-NEXT:       [[TOP3:%.*]] = inttoptr <4 x i64> [[EXT3]] to <4 x i32*>
; CHECK-NEXT:       ret void
  %interleaved.vec = load <12 x i32*>, <12 x i32*>* %ptr, align 4
  %v0 = shufflevector <12 x i32*> %interleaved.vec, <12 x i32*> poison, <4 x i32> <i32 0, i32 3, i32 6, i32 9>
  %v1 = shufflevector <12 x i32*> %interleaved.vec, <12 x i32*> poison, <4 x i32> <i32 1, i32 4, i32 7, i32 10>
  %v2 = shufflevector <12 x i32*> %interleaved.vec, <12 x i32*> poison, <4 x i32> <i32 2, i32 5, i32 8, i32 11>
  ret void
}

define void @load_ptrvec_factor4(<16 x i32*>* %ptr) #0 {
; CHECK-LABEL:    @load_ptrvec_factor4(
; CHECK-NEXT:       [[PTRUE:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
; CHECK-NEXT:       [[TMP1:%.*]] = bitcast <16 x i32*>* %ptr to i64*
; CHECK-NEXT:       [[LDN:%.*]] = call { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.ld4.sret.nxv2i64(<vscale x 2 x i1> [[PTRUE]], i64* [[TMP1]])
; CHECK-NEXT:       [[TMP2:%.*]] = extractvalue { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } [[LDN]], 3
; CHECK-NEXT:       [[EXT1:%.*]] = call <4 x i64> @llvm.vector.extract.v4i64.nxv2i64(<vscale x 2 x i64> [[TMP2]], i64 0)
; CHECK-NEXT:       [[TOP1:%.*]] = inttoptr <4 x i64> [[EXT1]] to <4 x i32*>
; CHECK-NEXT:       [[TMP3:%.*]] = extractvalue { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } [[LDN]], 2
; CHECK-NEXT:       [[EXT2:%.*]] = call <4 x i64> @llvm.vector.extract.v4i64.nxv2i64(<vscale x 2 x i64> [[TMP3]], i64 0)
; CHECK-NEXT:       [[TOP2:%.*]] = inttoptr <4 x i64> [[EXT2]] to <4 x i32*>
; CHECK-NEXT:       [[TMP4:%.*]] = extractvalue { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } [[LDN]], 1
; CHECK-NEXT:       [[EXT3:%.*]] = call <4 x i64> @llvm.vector.extract.v4i64.nxv2i64(<vscale x 2 x i64> [[TMP4]], i64 0)
; CHECK-NEXT:       [[TOP3:%.*]] = inttoptr <4 x i64> [[EXT3]] to <4 x i32*>
; CHECK-NEXT:       [[TMP5:%.*]] = extractvalue { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } [[LDN]], 0
; CHECK-NEXT:       [[EXT4:%.*]] = call <4 x i64> @llvm.vector.extract.v4i64.nxv2i64(<vscale x 2 x i64> [[TMP5]], i64 0)
; CHECK-NEXT:       [[TOP4:%.*]] = inttoptr <4 x i64> [[EXT4]] to <4 x i32*>
; CHECK-NEXT:       ret void
  %interleaved.vec = load <16 x i32*>, <16 x i32*>* %ptr, align 4
  %v0 = shufflevector <16 x i32*> %interleaved.vec, <16 x i32*> poison, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %v1 = shufflevector <16 x i32*> %interleaved.vec, <16 x i32*> poison, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %v2 = shufflevector <16 x i32*> %interleaved.vec, <16 x i32*> poison, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %v3 = shufflevector <16 x i32*> %interleaved.vec, <16 x i32*> poison, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  ret void
}

define void @store_ptrvec_factor2(<8 x i32*>* %ptr, <4 x i32*> %v0, <4 x i32*> %v1) #0 {
; CHECK-LABEL:    @store_ptrvec_factor2(
; CHECK-NEXT:       [[TOI1:%.*]] = ptrtoint <4 x i32*> %v0 to <4 x i64>
; CHECK-NEXT:       [[TOI2:%.*]] = ptrtoint <4 x i32*> %v1 to <4 x i64>
; CHECK-NEXT:       [[PTRUE:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
; CHECK-NEXT:       [[TMP1:%.*]] = shufflevector <4 x i64> [[TOI1]], <4 x i64> [[TOI2]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT:       [[INS1:%.*]] = call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v4i64(<vscale x 2 x i64> undef, <4 x i64> [[TMP1]], i64 0)
; CHECK-NEXT:       [[TMP2:%.*]] = shufflevector <4 x i64> [[TOI1]], <4 x i64> [[TOI2]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; CHECK-NEXT:       [[INS2:%.*]] = call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v4i64(<vscale x 2 x i64> undef, <4 x i64> [[TMP2]], i64 0)
; CHECK-NEXT:       [[PTR:%.*]] = bitcast <8 x i32*>* %ptr to i64*
; CHECK-NEXT:       call void @llvm.aarch64.sve.st2.nxv2i64(<vscale x 2 x i64> [[INS1]], <vscale x 2 x i64> [[INS2]], <vscale x 2 x i1> [[PTRUE]], i64* [[PTR]])
; CHECK-NEXT:       ret void
  %interleaved.vec = shufflevector <4 x i32*> %v0, <4 x i32*> %v1, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x i32*> %interleaved.vec, <8 x i32*>* %ptr, align 4
  ret void
}

define void @store_ptrvec_factor3(<12 x i32*>* %ptr, <4 x i32*> %v0, <4 x i32*> %v1, <4 x i32*> %v2) #0 {
; CHECK-LABEL:    @store_ptrvec_factor3(
; CHECK:            [[TOI1:%.*]] = ptrtoint <8 x i32*> %s0 to <8 x i64>
; CHECK-NEXT:       [[TOI2:%.*]] = ptrtoint <8 x i32*> %s1 to <8 x i64>
; CHECK-NEXT:       [[PTRUE:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
; CHECK-NEXT:       [[TMP1:%.*]] = shufflevector <8 x i64> [[TOI1]], <8 x i64> [[TOI2]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT:       [[INS1:%.*]] = call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v4i64(<vscale x 2 x i64> undef, <4 x i64> [[TMP1]], i64 0)
; CHECK-NEXT:       [[TMP2:%.*]] = shufflevector <8 x i64> [[TOI1]], <8 x i64> [[TOI2]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; CHECK-NEXT:       [[INS2:%.*]] = call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v4i64(<vscale x 2 x i64> undef, <4 x i64> [[TMP2]], i64 0)
; CHECK-NEXT:       [[TMP3:%.*]] = shufflevector <8 x i64> [[TOI1]], <8 x i64> [[TOI2]], <4 x i32> <i32 8, i32 9, i32 10, i32 11>
; CHECK-NEXT:       [[INS3:%.*]] = call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v4i64(<vscale x 2 x i64> undef, <4 x i64> [[TMP3]], i64 0)
; CHECK-NEXT:       [[PTR:%.*]] = bitcast <12 x i32*>* %ptr to i64*
; CHECK-NEXT:       call void @llvm.aarch64.sve.st3.nxv2i64(<vscale x 2 x i64> [[INS1]], <vscale x 2 x i64> [[INS2]], <vscale x 2 x i64> [[INS3]], <vscale x 2 x i1> [[PTRUE]], i64* [[PTR]])
; CHECK-NEXT:       ret void
  %s0 = shufflevector <4 x i32*> %v0, <4 x i32*> %v1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %s1 = shufflevector <4 x i32*> %v2, <4 x i32*> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  %interleaved.vec = shufflevector <8 x i32*> %s0, <8 x i32*> %s1, <12 x i32> <i32 0, i32 4, i32 8, i32 1, i32 5, i32 9, i32 2, i32 6, i32 10, i32 3, i32 7, i32 11>
  store <12 x i32*> %interleaved.vec, <12 x i32*>* %ptr, align 4
  ret void
}

define void @store_ptrvec_factor4(<16 x i32*>* %ptr, <4 x i32*> %v0, <4 x i32*> %v1, <4 x i32*> %v2, <4 x i32*> %v3) #0 {
; CHECK-LABEL:    @store_ptrvec_factor4(
; CHECK:            [[TOI1:%.*]] = ptrtoint <8 x i32*> %s0 to <8 x i64>
; CHECK-NEXT:       [[TOI2:%.*]] = ptrtoint <8 x i32*> %s1 to <8 x i64>
; CHECK-NEXT:       [[PTRUE:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
; CHECK-NEXT:       [[TMP1:%.*]] = shufflevector <8 x i64> [[TOI1]], <8 x i64> [[TOI2]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT:       [[INS1:%.*]] = call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v4i64(<vscale x 2 x i64> undef, <4 x i64> [[TMP1]], i64 0)
; CHECK-NEXT:       [[TMP2:%.*]] = shufflevector <8 x i64> [[TOI1]], <8 x i64> [[TOI2]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; CHECK-NEXT:       [[INS2:%.*]] = call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v4i64(<vscale x 2 x i64> undef, <4 x i64> [[TMP2]], i64 0)
; CHECK-NEXT:       [[TMP3:%.*]] = shufflevector <8 x i64> [[TOI1]], <8 x i64> [[TOI2]], <4 x i32> <i32 8, i32 9, i32 10, i32 11>
; CHECK-NEXT:       [[INS3:%.*]] = call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v4i64(<vscale x 2 x i64> undef, <4 x i64> [[TMP3]], i64 0)
; CHECK-NEXT:       [[TMP4:%.*]] = shufflevector <8 x i64> [[TOI1]], <8 x i64> [[TOI2]], <4 x i32> <i32 12, i32 13, i32 14, i32 15>
; CHECK-NEXT:       [[INS4:%.*]] = call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v4i64(<vscale x 2 x i64> undef, <4 x i64> [[TMP4]], i64 0)
; CHECK-NEXT:       [[PTR:%.*]] = bitcast <16 x i32*>* %ptr to i64*
; CHECK-NEXT:       call void @llvm.aarch64.sve.st4.nxv2i64(<vscale x 2 x i64> [[INS1]], <vscale x 2 x i64> [[INS2]], <vscale x 2 x i64> [[INS3]], <vscale x 2 x i64> [[INS4]], <vscale x 2 x i1> [[PTRUE]], i64* [[PTR]])
; CHECK-NEXT:       ret void
  %s0 = shufflevector <4 x i32*> %v0, <4 x i32*> %v1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %s1 = shufflevector <4 x i32*> %v2, <4 x i32*> %v3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %interleaved.vec = shufflevector <8 x i32*> %s0, <8 x i32*> %s1, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13,
                                                                               i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  store <16 x i32*> %interleaved.vec, <16 x i32*>* %ptr, align 4
  ret void
}

define void @load_factor2_wide(<16 x i64>* %ptr) #0 {
; CHECK-LABEL:    @load_factor2_wide(
; CHECK-NEXT:       [[TMP1:%.*]] = bitcast <16 x i64>* %ptr to i64*
; CHECK-NEXT:       [[PTRUE:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
; CHECK-NEXT:       [[LDN:%.*]] = call { <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.ld2.sret.nxv2i64(<vscale x 2 x i1> [[PTRUE]], i64* [[TMP1]])
; CHECK-NEXT:       [[TMP2:%.*]] = extractvalue { <vscale x 2 x i64>, <vscale x 2 x i64> } [[LDN]], 1
; CHECK-NEXT:       [[EXT1:%.*]] = call <4 x i64> @llvm.vector.extract.v4i64.nxv2i64(<vscale x 2 x i64> [[TMP2]], i64 0)
; CHECK-NEXT:       [[TMP3:%.*]] = extractvalue { <vscale x 2 x i64>, <vscale x 2 x i64> } [[LDN]], 0
; CHECK-NEXT:       [[EXT2:%.*]] = call <4 x i64> @llvm.vector.extract.v4i64.nxv2i64(<vscale x 2 x i64> [[TMP3]], i64 0)
; CHECK-NEXT:       [[TMP4:%.*]] = getelementptr i64, i64* [[TMP1]], i32 8
; CHECK-NEXT:       [[LDN:%.*]] = call { <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.ld2.sret.nxv2i64(<vscale x 2 x i1> [[PTRUE]], i64* [[TMP4]])
; CHECK-NEXT:       [[TMP5:%.*]] = extractvalue { <vscale x 2 x i64>, <vscale x 2 x i64> } [[LDN]], 1
; CHECK-NEXT:       [[EXT3:%.*]] = call <4 x i64> @llvm.vector.extract.v4i64.nxv2i64(<vscale x 2 x i64> [[TMP5]], i64 0)
; CHECK-NEXT:       [[TMP6:%.*]] = extractvalue { <vscale x 2 x i64>, <vscale x 2 x i64> } [[LDN]], 0
; CHECK-NEXT:       [[EXT4:%.*]] = call <4 x i64> @llvm.vector.extract.v4i64.nxv2i64(<vscale x 2 x i64> [[TMP6]], i64 0)
; CHECK-NEXT:       [[TMP7:%.*]] = shufflevector <4 x i64> [[EXT1]], <4 x i64> [[EXT3]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK-NEXT:       [[TMP8:%.*]] = shufflevector <4 x i64> [[EXT2]], <4 x i64> [[EXT4]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK-NEXT:       ret void
  %interleaved.vec = load <16 x i64>, <16 x i64>* %ptr, align 4
  %v0 = shufflevector <16 x i64> %interleaved.vec, <16 x i64> poison, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %v1 = shufflevector <16 x i64> %interleaved.vec, <16 x i64> poison, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  ret void
}

define void @store_factor2_wide(<16 x i64>* %ptr, <8 x i64> %v0, <8 x i64> %v1) #0 {
; CHECK-LABEL:    @store_factor2_wide(
; CHECK-NEXT:       [[TMP1:%.*]] = bitcast <16 x i64>* %ptr to i64*
; CHECK-NEXT:       [[PTRUE:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
; CHECK-NEXT:       [[TMP2:%.*]] = shufflevector <8 x i64> %v0, <8 x i64> %v1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT:       [[INS1:%.*]] = call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v4i64(<vscale x 2 x i64> undef, <4 x i64> [[TMP2]], i64 0)
; CHECK-NEXT:       [[TMP3:%.*]] = shufflevector <8 x i64> %v0, <8 x i64> %v1, <4 x i32> <i32 8, i32 9, i32 10, i32 11>
; CHECK-NEXT:       [[INS2:%.*]] = call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v4i64(<vscale x 2 x i64> undef, <4 x i64> [[TMP3]], i64 0)
; CHECK-NEXT:       call void @llvm.aarch64.sve.st2.nxv2i64(<vscale x 2 x i64> [[INS1]], <vscale x 2 x i64> [[INS2]], <vscale x 2 x i1> [[PTRUE]], i64* [[TMP1]])
; CHECK-NEXT:       [[TMP4:%.*]] = shufflevector <8 x i64> %v0, <8 x i64> %v1, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; CHECK-NEXT:       [[INS3:%.*]] = call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v4i64(<vscale x 2 x i64> undef, <4 x i64> [[TMP4]], i64 0)
; CHECK-NEXT:       [[TMP5:%.*]] = shufflevector <8 x i64> %v0, <8 x i64> %v1, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
; CHECK-NEXT:       [[INS4:%.*]] = call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v4i64(<vscale x 2 x i64> undef, <4 x i64> [[TMP5]], i64 0)
; CHECK-NEXT:       [[TMP6:%.*]] = getelementptr i64, i64* [[TMP1]], i32 8
; CHECK-NEXT:       call void @llvm.aarch64.sve.st2.nxv2i64(<vscale x 2 x i64> [[INS3]], <vscale x 2 x i64> [[INS4]], <vscale x 2 x i1> [[PTRUE]], i64* [[TMP6]])
; CHECK-NEXT:       ret void
  %interleaved.vec = shufflevector <8 x i64> %v0, <8 x i64> %v1, <16 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11, i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  store <16 x i64> %interleaved.vec, <16 x i64>* %ptr, align 4
  ret void
}

; Check that neon is used for illegal multiples of 128-bit types
define void @load_384bit(<12 x i64>* %ptr) #0 {
; CHECK-LABEL:    @load_384bit(
; CHECK: llvm.aarch64.neon.ld2
; CHECK-NOT: llvm.aarch64.sve.ld2
  %interleaved.vec = load <12 x i64>, <12 x i64>* %ptr, align 4
  %v0 = shufflevector <12 x i64> %interleaved.vec, <12 x i64> poison, <6 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10>
  %v1 = shufflevector <12 x i64> %interleaved.vec, <12 x i64> poison, <6 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11>
  ret void
}

; Check that neon is used for 128-bit vectors
define void @load_128bit(<4 x i64>* %ptr) #0 {
; CHECK-LABEL:    @load_128bit(
; CHECK: llvm.aarch64.neon.ld2
; CHECK-NOT: llvm.aarch64.sve.ld2
  %interleaved.vec = load <4 x i64>, <4 x i64>* %ptr, align 4
  %v0 = shufflevector <4 x i64> %interleaved.vec, <4 x i64> poison, <2 x i32> <i32 0, i32 2>
  %v1 = shufflevector <4 x i64> %interleaved.vec, <4 x i64> poison, <2 x i32> <i32 1, i32 3>
  ret void
}

; Check that correct ptrues are generated for min != max case
define void @load_min_not_max(<8 x i64>* %ptr) #1 {
; CHECK-LABEL:    @load_min_not_max(
; CHECK: call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 4)
  %interleaved.vec = load <8 x i64>, <8 x i64>* %ptr, align 4
  %v0 = shufflevector <8 x i64> %interleaved.vec, <8 x i64> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %v1 = shufflevector <8 x i64> %interleaved.vec, <8 x i64> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  ret void
}

define void @store_min_not_max(<8 x i64>* %ptr, <4 x i64> %v0, <4 x i64> %v1) #1 {
; CHECK-LABEL:    @store_min_not_max(
; CHECK: call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 4)
  %interleaved.vec = shufflevector <4 x i64> %v0, <4 x i64> %v1, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x i64> %interleaved.vec, <8 x i64>* %ptr, align 4
  ret void
}

; Check that correct ptrues are generated for min > type case
define void @load_min_ge_type(<8 x i64>* %ptr) #2 {
; CHECK-LABEL:    @load_min_ge_type(
; CHECK: call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 4)
  %interleaved.vec = load <8 x i64>, <8 x i64>* %ptr, align 4
  %v0 = shufflevector <8 x i64> %interleaved.vec, <8 x i64> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %v1 = shufflevector <8 x i64> %interleaved.vec, <8 x i64> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  ret void
}

define void @store_min_ge_type(<8 x i64>* %ptr, <4 x i64> %v0, <4 x i64> %v1) #2 {
; CHECK-LABEL:    @store_min_ge_type(
; CHECK: call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 4)
  %interleaved.vec = shufflevector <4 x i64> %v0, <4 x i64> %v1, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x i64> %interleaved.vec, <8 x i64>* %ptr, align 4
  ret void
}

define void @load_double_factor4(<16 x double>* %ptr) #0 {
; CHECK-LABEL: @load_double_factor4(
; CHECK-NEXT:    [[TMP1:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast <16 x double>* [[PTR:%.*]] to double*
; CHECK-NEXT:    [[LDN:%.*]] = call { <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double> } @llvm.aarch64.sve.ld4.sret.nxv2f64(<vscale x 2 x i1> [[TMP1]], double* [[TMP2]])
; CHECK-NEXT:    [[TMP3:%.*]] = extractvalue { <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double> } [[LDN]], 3
; CHECK-NEXT:    [[TMP4:%.*]] = call <4 x double> @llvm.vector.extract.v4f64.nxv2f64(<vscale x 2 x double> [[TMP3]], i64 0)
; CHECK-NEXT:    [[TMP5:%.*]] = extractvalue { <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double> } [[LDN]], 2
; CHECK-NEXT:    [[TMP6:%.*]] = call <4 x double> @llvm.vector.extract.v4f64.nxv2f64(<vscale x 2 x double> [[TMP5]], i64 0)
; CHECK-NEXT:    [[TMP7:%.*]] = extractvalue { <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double> } [[LDN]], 1
; CHECK-NEXT:    [[TMP8:%.*]] = call <4 x double> @llvm.vector.extract.v4f64.nxv2f64(<vscale x 2 x double> [[TMP7]], i64 0)
; CHECK-NEXT:    [[TMP9:%.*]] = extractvalue { <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double> } [[LDN]], 0
; CHECK-NEXT:    [[TMP10:%.*]] = call <4 x double> @llvm.vector.extract.v4f64.nxv2f64(<vscale x 2 x double> [[TMP9]], i64 0)
; CHECK-NEXT:    ret void
;
  %interleaved.vec = load <16 x double>, <16 x double>* %ptr, align 4
  %v0 = shufflevector <16 x double> %interleaved.vec, <16 x double> poison, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %v1 = shufflevector <16 x double> %interleaved.vec, <16 x double> poison, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %v2 = shufflevector <16 x double> %interleaved.vec, <16 x double> poison, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %v3 = shufflevector <16 x double> %interleaved.vec, <16 x double> poison, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  ret void
}

define void @load_float_factor3(<24 x float>* %ptr) #0 {
; CHECK-LABEL: @load_float_factor3(
; CHECK-NEXT:    [[TMP1:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast <24 x float>* [[PTR:%.*]] to float*
; CHECK-NEXT:    [[LDN:%.*]] = call { <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float> } @llvm.aarch64.sve.ld3.sret.nxv4f32(<vscale x 4 x i1> [[TMP1]], float* [[TMP2]])
; CHECK-NEXT:    [[TMP3:%.*]] = extractvalue { <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float> } [[LDN]], 2
; CHECK-NEXT:    [[TMP4:%.*]] = call <8 x float> @llvm.vector.extract.v8f32.nxv4f32(<vscale x 4 x float> [[TMP3]], i64 0)
; CHECK-NEXT:    [[TMP5:%.*]] = extractvalue { <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float> } [[LDN]], 1
; CHECK-NEXT:    [[TMP6:%.*]] = call <8 x float> @llvm.vector.extract.v8f32.nxv4f32(<vscale x 4 x float> [[TMP5]], i64 0)
; CHECK-NEXT:    [[TMP7:%.*]] = extractvalue { <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float> } [[LDN]], 0
; CHECK-NEXT:    [[TMP8:%.*]] = call <8 x float> @llvm.vector.extract.v8f32.nxv4f32(<vscale x 4 x float> [[TMP7]], i64 0)
; CHECK-NEXT:    ret void
;
  %interleaved.vec = load <24 x float>, <24 x float>* %ptr, align 4
  %v0 = shufflevector <24 x float> %interleaved.vec, <24 x float> poison, <8 x i32> <i32 0, i32 3, i32 6, i32 9, i32 12, i32 15, i32 18, i32 21>
  %v1 = shufflevector <24 x float> %interleaved.vec, <24 x float> poison, <8 x i32> <i32 1, i32 4, i32 7, i32 10, i32 13, i32 16, i32 19, i32 22>
  %v2 = shufflevector <24 x float> %interleaved.vec, <24 x float> poison, <8 x i32> <i32 2, i32 5, i32 8, i32 11, i32 14, i32 17, i32 20, i32 23>
  ret void
}

define void @load_half_factor2(<32 x half>* %ptr) #0 {
; CHECK-LABEL: @load_half_factor2(
; CHECK-NEXT:    [[TMP1:%.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast <32 x half>* [[PTR:%.*]] to half*
; CHECK-NEXT:    [[LDN:%.*]] = call { <vscale x 8 x half>, <vscale x 8 x half> } @llvm.aarch64.sve.ld2.sret.nxv8f16(<vscale x 8 x i1> [[TMP1]], half* [[TMP2]])
; CHECK-NEXT:    [[TMP3:%.*]] = extractvalue { <vscale x 8 x half>, <vscale x 8 x half> } [[LDN]], 1
; CHECK-NEXT:    [[TMP4:%.*]] = call <16 x half> @llvm.vector.extract.v16f16.nxv8f16(<vscale x 8 x half> [[TMP3]], i64 0)
; CHECK-NEXT:    [[TMP5:%.*]] = extractvalue { <vscale x 8 x half>, <vscale x 8 x half> } [[LDN]], 0
; CHECK-NEXT:    [[TMP6:%.*]] = call <16 x half> @llvm.vector.extract.v16f16.nxv8f16(<vscale x 8 x half> [[TMP5]], i64 0)
; CHECK-NEXT:    ret void
;
  %interleaved.vec = load <32 x half>, <32 x half>* %ptr, align 4
  %v0 = shufflevector <32 x half> %interleaved.vec, <32 x half> poison, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %v1 = shufflevector <32 x half> %interleaved.vec, <32 x half> poison, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  ret void
}

define void @load_bfloat_factor2(<32 x bfloat>* %ptr) #0 {
; CHECK-LABEL: @load_bfloat_factor2(
; CHECK-NEXT:    [[TMP1:%.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast <32 x bfloat>* [[PTR:%.*]] to bfloat*
; CHECK-NEXT:    [[LDN:%.*]] = call { <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } @llvm.aarch64.sve.ld2.sret.nxv8bf16(<vscale x 8 x i1> [[TMP1]], bfloat* [[TMP2]])
; CHECK-NEXT:    [[TMP3:%.*]] = extractvalue { <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } [[LDN]], 1
; CHECK-NEXT:    [[TMP4:%.*]] = call <16 x bfloat> @llvm.vector.extract.v16bf16.nxv8bf16(<vscale x 8 x bfloat> [[TMP3]], i64 0)
; CHECK-NEXT:    [[TMP5:%.*]] = extractvalue { <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } [[LDN]], 0
; CHECK-NEXT:    [[TMP6:%.*]] = call <16 x bfloat> @llvm.vector.extract.v16bf16.nxv8bf16(<vscale x 8 x bfloat> [[TMP5]], i64 0)
; CHECK-NEXT:    ret void
;
  %interleaved.vec = load <32 x bfloat>, <32 x bfloat>* %ptr, align 4
  %v0 = shufflevector <32 x bfloat> %interleaved.vec, <32 x bfloat> poison, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %v1 = shufflevector <32 x bfloat> %interleaved.vec, <32 x bfloat> poison, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  ret void
}

define void @store_double_factor4(<16 x double>* %ptr, <4 x double> %v0, <4 x double> %v1, <4 x double> %v2, <4 x double> %v3) #0 {
; CHECK-LABEL: @store_double_factor4(
; CHECK-NEXT:    [[S0:%.*]] = shufflevector <4 x double> [[V0:%.*]], <4 x double> [[V1:%.*]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK-NEXT:    [[S1:%.*]] = shufflevector <4 x double> [[V2:%.*]], <4 x double> [[V3:%.*]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK-NEXT:    [[TMP1:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
; CHECK-NEXT:    [[TMP2:%.*]] = shufflevector <8 x double> [[S0]], <8 x double> [[S1]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT:    [[TMP3:%.*]] = call <vscale x 2 x double> @llvm.vector.insert.nxv2f64.v4f64(<vscale x 2 x double> undef, <4 x double> [[TMP2]], i64 0)
; CHECK-NEXT:    [[TMP4:%.*]] = shufflevector <8 x double> [[S0]], <8 x double> [[S1]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; CHECK-NEXT:    [[TMP5:%.*]] = call <vscale x 2 x double> @llvm.vector.insert.nxv2f64.v4f64(<vscale x 2 x double> undef, <4 x double> [[TMP4]], i64 0)
; CHECK-NEXT:    [[TMP6:%.*]] = shufflevector <8 x double> [[S0]], <8 x double> [[S1]], <4 x i32> <i32 8, i32 9, i32 10, i32 11>
; CHECK-NEXT:    [[TMP7:%.*]] = call <vscale x 2 x double> @llvm.vector.insert.nxv2f64.v4f64(<vscale x 2 x double> undef, <4 x double> [[TMP6]], i64 0)
; CHECK-NEXT:    [[TMP8:%.*]] = shufflevector <8 x double> [[S0]], <8 x double> [[S1]], <4 x i32> <i32 12, i32 13, i32 14, i32 15>
; CHECK-NEXT:    [[TMP9:%.*]] = call <vscale x 2 x double> @llvm.vector.insert.nxv2f64.v4f64(<vscale x 2 x double> undef, <4 x double> [[TMP8]], i64 0)
; CHECK-NEXT:    [[TMP10:%.*]] = bitcast <16 x double>* [[PTR:%.*]] to double*
; CHECK-NEXT:    call void @llvm.aarch64.sve.st4.nxv2f64(<vscale x 2 x double> [[TMP3]], <vscale x 2 x double> [[TMP5]], <vscale x 2 x double> [[TMP7]], <vscale x 2 x double> [[TMP9]], <vscale x 2 x i1> [[TMP1]], double* [[TMP10]])
; CHECK-NEXT:    ret void
;
  %s0 = shufflevector <4 x double> %v0, <4 x double> %v1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %s1 = shufflevector <4 x double> %v2, <4 x double> %v3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %interleaved.vec = shufflevector <8 x double> %s0, <8 x double> %s1, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  store <16 x double> %interleaved.vec, <16 x double>* %ptr, align 4
  ret void
}

define void @store_float_factor3(<24 x float>* %ptr, <8 x float> %v0, <8 x float> %v1, <8 x float> %v2) #0 {
; CHECK-LABEL: @store_float_factor3(
; CHECK-NEXT:    [[S0:%.*]] = shufflevector <8 x float> [[V0:%.*]], <8 x float> [[V1:%.*]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
; CHECK-NEXT:    [[S1:%.*]] = shufflevector <8 x float> [[V2:%.*]], <8 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:    [[TMP1:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
; CHECK-NEXT:    [[TMP2:%.*]] = shufflevector <16 x float> [[S0]], <16 x float> [[S1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK-NEXT:    [[TMP3:%.*]] = call <vscale x 4 x float> @llvm.vector.insert.nxv4f32.v8f32(<vscale x 4 x float> undef, <8 x float> [[TMP2]], i64 0)
; CHECK-NEXT:    [[TMP4:%.*]] = shufflevector <16 x float> [[S0]], <16 x float> [[S1]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
; CHECK-NEXT:    [[TMP5:%.*]] = call <vscale x 4 x float> @llvm.vector.insert.nxv4f32.v8f32(<vscale x 4 x float> undef, <8 x float> [[TMP4]], i64 0)
; CHECK-NEXT:    [[TMP6:%.*]] = shufflevector <16 x float> [[S0]], <16 x float> [[S1]], <8 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
; CHECK-NEXT:    [[TMP7:%.*]] = call <vscale x 4 x float> @llvm.vector.insert.nxv4f32.v8f32(<vscale x 4 x float> undef, <8 x float> [[TMP6]], i64 0)
; CHECK-NEXT:    [[TMP8:%.*]] = bitcast <24 x float>* [[PTR:%.*]] to float*
; CHECK-NEXT:    call void @llvm.aarch64.sve.st3.nxv4f32(<vscale x 4 x float> [[TMP3]], <vscale x 4 x float> [[TMP5]], <vscale x 4 x float> [[TMP7]], <vscale x 4 x i1> [[TMP1]], float* [[TMP8]])
; CHECK-NEXT:    ret void
;
  %s0 = shufflevector <8 x float> %v0, <8 x float> %v1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
  i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %s1 = shufflevector <8 x float> %v2, <8 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
  i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %interleaved.vec = shufflevector <16 x float> %s0, <16 x float> %s1, <24 x i32> <i32 0, i32 8, i32 16, i32 1, i32 9, i32 17, i32 2, i32 10, i32 18, i32 3, i32 11, i32 19,
  i32 4, i32 12, i32 20, i32 5, i32 13, i32 21, i32 6, i32 14, i32 22, i32 7, i32 15, i32 23>
  store <24 x float> %interleaved.vec, <24 x float>* %ptr, align 4
  ret void
}

define void @store_half_factor2(<32 x half>* %ptr, <16 x half> %v0, <16 x half> %v1) #0 {
; CHECK-LABEL: @store_half_factor2(
; CHECK-NEXT:    [[TMP1:%.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
; CHECK-NEXT:    [[TMP2:%.*]] = shufflevector <16 x half> [[V0:%.*]], <16 x half> [[V1:%.*]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
; CHECK-NEXT:    [[TMP3:%.*]] = call <vscale x 8 x half> @llvm.vector.insert.nxv8f16.v16f16(<vscale x 8 x half> undef, <16 x half> [[TMP2]], i64 0)
; CHECK-NEXT:    [[TMP4:%.*]] = shufflevector <16 x half> [[V0]], <16 x half> [[V1]], <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
; CHECK-NEXT:    [[TMP5:%.*]] = call <vscale x 8 x half> @llvm.vector.insert.nxv8f16.v16f16(<vscale x 8 x half> undef, <16 x half> [[TMP4]], i64 0)
; CHECK-NEXT:    [[TMP6:%.*]] = bitcast <32 x half>* [[PTR:%.*]] to half*
; CHECK-NEXT:    call void @llvm.aarch64.sve.st2.nxv8f16(<vscale x 8 x half> [[TMP3]], <vscale x 8 x half> [[TMP5]], <vscale x 8 x i1> [[TMP1]], half* [[TMP6]])
; CHECK-NEXT:    ret void
;
  %interleaved.vec = shufflevector <16 x half> %v0, <16 x half> %v1, <32 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23,
  i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  store <32 x half> %interleaved.vec, <32 x half>* %ptr, align 4
  ret void
}


define void @store_bfloat_factor2(<32 x bfloat>* %ptr, <16 x bfloat> %v0, <16 x bfloat> %v1) #0 {
; CHECK-LABEL: @store_bfloat_factor2(
; CHECK-NEXT:    [[TMP1:%.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
; CHECK-NEXT:    [[TMP2:%.*]] = shufflevector <16 x bfloat> [[V0:%.*]], <16 x bfloat> [[V1:%.*]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
; CHECK-NEXT:    [[TMP3:%.*]] = call <vscale x 8 x bfloat> @llvm.vector.insert.nxv8bf16.v16bf16(<vscale x 8 x bfloat> undef, <16 x bfloat> [[TMP2]], i64 0)
; CHECK-NEXT:    [[TMP4:%.*]] = shufflevector <16 x bfloat> [[V0]], <16 x bfloat> [[V1]], <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
; CHECK-NEXT:    [[TMP5:%.*]] = call <vscale x 8 x bfloat> @llvm.vector.insert.nxv8bf16.v16bf16(<vscale x 8 x bfloat> undef, <16 x bfloat> [[TMP4]], i64 0)
; CHECK-NEXT:    [[TMP6:%.*]] = bitcast <32 x bfloat>* [[PTR:%.*]] to bfloat*
; CHECK-NEXT:    call void @llvm.aarch64.sve.st2.nxv8bf16(<vscale x 8 x bfloat> [[TMP3]], <vscale x 8 x bfloat> [[TMP5]], <vscale x 8 x i1> [[TMP1]], bfloat* [[TMP6]])
; CHECK-NEXT:    ret void
;
  %interleaved.vec = shufflevector <16 x bfloat> %v0, <16 x bfloat> %v1, <32 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23,
  i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  store <32 x bfloat> %interleaved.vec, <32 x bfloat>* %ptr, align 4
  ret void
}

attributes #0 = { vscale_range(2,2) "target-features"="+sve" }
attributes #1 = { vscale_range(2,4) "target-features"="+sve" }
attributes #2 = { vscale_range(4,4) "target-features"="+sve" }
