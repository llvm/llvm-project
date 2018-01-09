; RUN: llc -mtriple=arm-eabi -mattr=+neon -print-before=post-RA-sched %s -o - 2>&1 \
; RUN:  | FileCheck %s

define void @vst(i8* %m, [4 x i64] %v) {
entry:
; CHECK: vst:
; CHECK: VST1d64Q killed %r{{[0-9]+}}, 8, %d{{[0-9]+}}, 14, %noreg, implicit killed %q{{[0-9]+}}_q{{[0-9]+}}

  %v0 = extractvalue [4 x i64] %v, 0
  %v1 = extractvalue [4 x i64] %v, 1
  %v2 = extractvalue [4 x i64] %v, 2
  %v3 = extractvalue [4 x i64] %v, 3

  %t0 = bitcast i64 %v0 to <8 x i8>
  %t1 = bitcast i64 %v1 to <8 x i8>
  %t2 = bitcast i64 %v2 to <8 x i8>
  %t3 = bitcast i64 %v3 to <8 x i8>

  %s0 = bitcast <8 x i8> %t0 to <1 x i64>
  %s1 = bitcast <8 x i8> %t1 to <1 x i64>
  %s2 = bitcast <8 x i8> %t2 to <1 x i64>
  %s3 = bitcast <8 x i8> %t3 to <1 x i64>

  %tmp0 = bitcast <1 x i64> %s2 to i64
  %tmp1 = bitcast <1 x i64> %s3 to i64

  %n0 = insertelement <2 x i64> undef, i64 %tmp0, i32 0
  %n1 = insertelement <2 x i64> %n0, i64 %tmp1, i32 1

  call void @llvm.arm.neon.vst4.p0i8.v1i64(i8* %m, <1 x i64> %s0, <1 x i64> %s1, <1 x i64> %s2, <1 x i64> %s3, i32 8)

  call void @bar(<2 x i64> %n1)

  ret void
}

%struct.__neon_int8x8x4_t = type { <8 x i8>,  <8 x i8>,  <8 x i8>, <8 x i8> }
define <8 x i8> @vtbx4(<8 x i8>* %A, %struct.__neon_int8x8x4_t* %B, <8 x i8>* %C) nounwind {
; CHECK: vtbx4:
; CHECK: VTBX4 {{.*}}, 14, %noreg, implicit %q{{[0-9]+}}_q{{[0-9]+}}
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load %struct.__neon_int8x8x4_t, %struct.__neon_int8x8x4_t* %B
        %tmp3 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 2
        %tmp6 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 3
	%tmp7 = load <8 x i8>, <8 x i8>* %C
	%tmp8 = call <8 x i8> @llvm.arm.neon.vtbx4(<8 x i8> %tmp1, <8 x i8> %tmp3, <8 x i8> %tmp4, <8 x i8> %tmp5, <8 x i8> %tmp6, <8 x i8> %tmp7)
  call void @bar2(%struct.__neon_int8x8x4_t %tmp2, <8 x i8> %tmp8)
	ret <8 x i8> %tmp8
}

declare void @llvm.arm.neon.vst4.p0i8.v1i64(i8*, <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64>, i32)
declare <8 x i8>  @llvm.arm.neon.vtbx4(<8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>) nounwind readnone
declare void @bar2(%struct.__neon_int8x8x4_t, <8 x i8>)
declare void @bar(<2 x i64> %arg)
