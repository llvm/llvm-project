; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s


define void @fcmltz_4s(<4 x float> %a, <4 x i16>* %p) nounwind {
;CHECK-LABEL: fcmltz_4s:
;CHECK: fcmlt.4s [[REG:v[0-9]+]], v0, #0
;CHECK-NEXT: xtn.4h v[[REG_1:[0-9]+]], [[REG]]
;CHECK-NEXT: str d[[REG_1]], [x0]
;CHECK-NEXT: ret
  %tmp = fcmp olt <4 x float> %a, zeroinitializer
  %tmp2 = sext <4 x i1> %tmp to <4 x i16>
  store <4 x i16> %tmp2, <4 x i16>* %p, align 8
  ret void
}

define <2 x i32> @facge_2s(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: facge_2s:
;CHECK: facge.2s
	%tmp1 = load <2 x float>, <2 x float>* %A
	%tmp2 = load <2 x float>, <2 x float>* %B
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.facge.v2i32.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x i32> %tmp3
}

define <4 x i32> @facge_4s(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK-LABEL: facge_4s:
;CHECK: facge.4s
	%tmp1 = load <4 x float>, <4 x float>* %A
	%tmp2 = load <4 x float>, <4 x float>* %B
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.facge.v4i32.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @facge_2d(<2 x double>* %A, <2 x double>* %B) nounwind {
;CHECK-LABEL: facge_2d:
;CHECK: facge.2d
	%tmp1 = load <2 x double>, <2 x double>* %A
	%tmp2 = load <2 x double>, <2 x double>* %B
	%tmp3 = call <2 x i64> @llvm.aarch64.neon.facge.v2i64.v2f64(<2 x double> %tmp1, <2 x double> %tmp2)
	ret <2 x i64> %tmp3
}

declare <2 x i32> @llvm.aarch64.neon.facge.v2i32.v2f32(<2 x float>, <2 x float>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.facge.v4i32.v4f32(<4 x float>, <4 x float>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.facge.v2i64.v2f64(<2 x double>, <2 x double>) nounwind readnone

define <2 x i32> @facgt_2s(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: facgt_2s:
;CHECK: facgt.2s
	%tmp1 = load <2 x float>, <2 x float>* %A
	%tmp2 = load <2 x float>, <2 x float>* %B
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.facgt.v2i32.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x i32> %tmp3
}

define <4 x i32> @facgt_4s(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK-LABEL: facgt_4s:
;CHECK: facgt.4s
	%tmp1 = load <4 x float>, <4 x float>* %A
	%tmp2 = load <4 x float>, <4 x float>* %B
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.facgt.v4i32.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @facgt_2d(<2 x double>* %A, <2 x double>* %B) nounwind {
;CHECK-LABEL: facgt_2d:
;CHECK: facgt.2d
	%tmp1 = load <2 x double>, <2 x double>* %A
	%tmp2 = load <2 x double>, <2 x double>* %B
	%tmp3 = call <2 x i64> @llvm.aarch64.neon.facgt.v2i64.v2f64(<2 x double> %tmp1, <2 x double> %tmp2)
	ret <2 x i64> %tmp3
}

declare <2 x i32> @llvm.aarch64.neon.facgt.v2i32.v2f32(<2 x float>, <2 x float>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.facgt.v4i32.v4f32(<4 x float>, <4 x float>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.facgt.v2i64.v2f64(<2 x double>, <2 x double>) nounwind readnone

define i32 @facge_s(float %A, float %B) nounwind {
; CHECK-LABEL: facge_s:
; CHECK: facge {{s[0-9]+}}, s0, s1
  %mask = call i32 @llvm.aarch64.neon.facge.i32.f32(float %A, float %B)
  ret i32 %mask
}

define i64 @facge_d(double %A, double %B) nounwind {
; CHECK-LABEL: facge_d:
; CHECK: facge {{d[0-9]+}}, d0, d1
  %mask = call i64 @llvm.aarch64.neon.facge.i64.f64(double %A, double %B)
  ret i64 %mask
}

declare i64 @llvm.aarch64.neon.facge.i64.f64(double, double)
declare i32 @llvm.aarch64.neon.facge.i32.f32(float, float)

define i32 @facgt_s(float %A, float %B) nounwind {
; CHECK-LABEL: facgt_s:
; CHECK: facgt {{s[0-9]+}}, s0, s1
  %mask = call i32 @llvm.aarch64.neon.facgt.i32.f32(float %A, float %B)
  ret i32 %mask
}

define i64 @facgt_d(double %A, double %B) nounwind {
; CHECK-LABEL: facgt_d:
; CHECK: facgt {{d[0-9]+}}, d0, d1
  %mask = call i64 @llvm.aarch64.neon.facgt.i64.f64(double %A, double %B)
  ret i64 %mask
}

declare i64 @llvm.aarch64.neon.facgt.i64.f64(double, double)
declare i32 @llvm.aarch64.neon.facgt.i32.f32(float, float)

define <8 x i8> @cmtst_8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: cmtst_8b:
;CHECK: cmtst.8b
  %tmp1 = load <8 x i8>, <8 x i8>* %A
  %tmp2 = load <8 x i8>, <8 x i8>* %B
  %commonbits = and <8 x i8> %tmp1, %tmp2
  %mask = icmp ne <8 x i8> %commonbits, zeroinitializer
  %res = sext <8 x i1> %mask to <8 x i8>
  ret <8 x i8> %res
}

define <16 x i8> @cmtst_16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: cmtst_16b:
;CHECK: cmtst.16b
  %tmp1 = load <16 x i8>, <16 x i8>* %A
  %tmp2 = load <16 x i8>, <16 x i8>* %B
  %commonbits = and <16 x i8> %tmp1, %tmp2
  %mask = icmp ne <16 x i8> %commonbits, zeroinitializer
  %res = sext <16 x i1> %mask to <16 x i8>
  ret <16 x i8> %res
}

define <4 x i16> @cmtst_4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: cmtst_4h:
;CHECK: cmtst.4h
  %tmp1 = load <4 x i16>, <4 x i16>* %A
  %tmp2 = load <4 x i16>, <4 x i16>* %B
  %commonbits = and <4 x i16> %tmp1, %tmp2
  %mask = icmp ne <4 x i16> %commonbits, zeroinitializer
  %res = sext <4 x i1> %mask to <4 x i16>
  ret <4 x i16> %res
}

define <8 x i16> @cmtst_8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: cmtst_8h:
;CHECK: cmtst.8h
  %tmp1 = load <8 x i16>, <8 x i16>* %A
  %tmp2 = load <8 x i16>, <8 x i16>* %B
  %commonbits = and <8 x i16> %tmp1, %tmp2
  %mask = icmp ne <8 x i16> %commonbits, zeroinitializer
  %res = sext <8 x i1> %mask to <8 x i16>
  ret <8 x i16> %res
}

define <2 x i32> @cmtst_2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: cmtst_2s:
;CHECK: cmtst.2s
  %tmp1 = load <2 x i32>, <2 x i32>* %A
  %tmp2 = load <2 x i32>, <2 x i32>* %B
  %commonbits = and <2 x i32> %tmp1, %tmp2
  %mask = icmp ne <2 x i32> %commonbits, zeroinitializer
  %res = sext <2 x i1> %mask to <2 x i32>
  ret <2 x i32> %res
}

define <4 x i32> @cmtst_4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: cmtst_4s:
;CHECK: cmtst.4s
  %tmp1 = load <4 x i32>, <4 x i32>* %A
  %tmp2 = load <4 x i32>, <4 x i32>* %B
  %commonbits = and <4 x i32> %tmp1, %tmp2
  %mask = icmp ne <4 x i32> %commonbits, zeroinitializer
  %res = sext <4 x i1> %mask to <4 x i32>
  ret <4 x i32> %res
}

define <2 x i64> @cmtst_2d(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: cmtst_2d:
;CHECK: cmtst.2d
  %tmp1 = load <2 x i64>, <2 x i64>* %A
  %tmp2 = load <2 x i64>, <2 x i64>* %B
  %commonbits = and <2 x i64> %tmp1, %tmp2
  %mask = icmp ne <2 x i64> %commonbits, zeroinitializer
  %res = sext <2 x i1> %mask to <2 x i64>
  ret <2 x i64> %res
}

define <1 x i64> @fcmeq_d(<1 x double> %A, <1 x double> %B) nounwind {
; CHECK-LABEL: fcmeq_d:
; CHECK: fcmeq {{d[0-9]+}}, d0, d1
  %tst = fcmp oeq <1 x double> %A, %B
  %mask = sext <1 x i1> %tst to <1 x i64>
  ret <1 x i64> %mask
}

define <1 x i64> @fcmge_d(<1 x double> %A, <1 x double> %B) nounwind {
; CHECK-LABEL: fcmge_d:
; CHECK: fcmge {{d[0-9]+}}, d0, d1
  %tst = fcmp oge <1 x double> %A, %B
  %mask = sext <1 x i1> %tst to <1 x i64>
  ret <1 x i64> %mask
}

define <1 x i64> @fcmle_d(<1 x double> %A, <1 x double> %B) nounwind {
; CHECK-LABEL: fcmle_d:
; CHECK: fcmge {{d[0-9]+}}, d1, d0
  %tst = fcmp ole <1 x double> %A, %B
  %mask = sext <1 x i1> %tst to <1 x i64>
  ret <1 x i64> %mask
}

define <1 x i64> @fcmgt_d(<1 x double> %A, <1 x double> %B) nounwind {
; CHECK-LABEL: fcmgt_d:
; CHECK: fcmgt {{d[0-9]+}}, d0, d1
  %tst = fcmp ogt <1 x double> %A, %B
  %mask = sext <1 x i1> %tst to <1 x i64>
  ret <1 x i64> %mask
}

define <1 x i64> @fcmlt_d(<1 x double> %A, <1 x double> %B) nounwind {
; CHECK-LABEL: fcmlt_d:
; CHECK: fcmgt {{d[0-9]+}}, d1, d0
  %tst = fcmp olt <1 x double> %A, %B
  %mask = sext <1 x i1> %tst to <1 x i64>
  ret <1 x i64> %mask
}

define <1 x i64> @cmnez_d(<1 x i64> %A) nounwind {
; CHECK-LABEL: cmnez_d:
; CHECK: cmeq d[[EQ:[0-9]+]], d0, #0
; CHECK: mvn.8b v0, v[[EQ]]
  %tst = icmp ne <1 x i64> %A, zeroinitializer
  %mask = sext <1 x i1> %tst to <1 x i64>
  ret <1 x i64> %mask
}
