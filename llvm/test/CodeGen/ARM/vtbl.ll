; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

%struct.__neon_int8x8x2_t = type { <8 x i8>, <8 x i8> }
%struct.__neon_int8x8x3_t = type { <8 x i8>,  <8 x i8>, <8 x i8> }
%struct.__neon_int8x8x4_t = type { <8 x i8>,  <8 x i8>,  <8 x i8>, <8 x i8> }

define <8 x i8> @vtbl1(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vtbl1:
;CHECK: vtbl.8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vtbl1(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <8 x i8> @vtbl2(<8 x i8>* %A, %struct.__neon_int8x8x2_t* %B) nounwind {
;CHECK: vtbl2:
;CHECK: vtbl.8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load %struct.__neon_int8x8x2_t* %B
        %tmp3 = extractvalue %struct.__neon_int8x8x2_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int8x8x2_t %tmp2, 1
	%tmp5 = call <8 x i8> @llvm.arm.neon.vtbl2(<8 x i8> %tmp1, <8 x i8> %tmp3, <8 x i8> %tmp4)
	ret <8 x i8> %tmp5
}

define <8 x i8> @vtbl3(<8 x i8>* %A, %struct.__neon_int8x8x3_t* %B) nounwind {
;CHECK: vtbl3:
;CHECK: vtbl.8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load %struct.__neon_int8x8x3_t* %B
        %tmp3 = extractvalue %struct.__neon_int8x8x3_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int8x8x3_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int8x8x3_t %tmp2, 2
	%tmp6 = call <8 x i8> @llvm.arm.neon.vtbl3(<8 x i8> %tmp1, <8 x i8> %tmp3, <8 x i8> %tmp4, <8 x i8> %tmp5)
	ret <8 x i8> %tmp6
}

define <8 x i8> @vtbl4(<8 x i8>* %A, %struct.__neon_int8x8x4_t* %B) nounwind {
;CHECK: vtbl4:
;CHECK: vtbl.8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load %struct.__neon_int8x8x4_t* %B
        %tmp3 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 2
        %tmp6 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 3
	%tmp7 = call <8 x i8> @llvm.arm.neon.vtbl4(<8 x i8> %tmp1, <8 x i8> %tmp3, <8 x i8> %tmp4, <8 x i8> %tmp5, <8 x i8> %tmp6)
	ret <8 x i8> %tmp7
}

define <8 x i8> @vtbx1(<8 x i8>* %A, <8 x i8>* %B, <8 x i8>* %C) nounwind {
;CHECK: vtbx1:
;CHECK: vtbx.8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
	%tmp4 = call <8 x i8> @llvm.arm.neon.vtbx1(<8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i8> %tmp3)
	ret <8 x i8> %tmp4
}

define <8 x i8> @vtbx2(<8 x i8>* %A, %struct.__neon_int8x8x2_t* %B, <8 x i8>* %C) nounwind {
;CHECK: vtbx2:
;CHECK: vtbx.8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load %struct.__neon_int8x8x2_t* %B
        %tmp3 = extractvalue %struct.__neon_int8x8x2_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int8x8x2_t %tmp2, 1
	%tmp5 = load <8 x i8>* %C
	%tmp6 = call <8 x i8> @llvm.arm.neon.vtbx2(<8 x i8> %tmp1, <8 x i8> %tmp3, <8 x i8> %tmp4, <8 x i8> %tmp5)
	ret <8 x i8> %tmp6
}

define <8 x i8> @vtbx3(<8 x i8>* %A, %struct.__neon_int8x8x3_t* %B, <8 x i8>* %C) nounwind {
;CHECK: vtbx3:
;CHECK: vtbx.8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load %struct.__neon_int8x8x3_t* %B
        %tmp3 = extractvalue %struct.__neon_int8x8x3_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int8x8x3_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int8x8x3_t %tmp2, 2
	%tmp6 = load <8 x i8>* %C
	%tmp7 = call <8 x i8> @llvm.arm.neon.vtbx3(<8 x i8> %tmp1, <8 x i8> %tmp3, <8 x i8> %tmp4, <8 x i8> %tmp5, <8 x i8> %tmp6)
	ret <8 x i8> %tmp7
}

define <8 x i8> @vtbx4(<8 x i8>* %A, %struct.__neon_int8x8x4_t* %B, <8 x i8>* %C) nounwind {
;CHECK: vtbx4:
;CHECK: vtbx.8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load %struct.__neon_int8x8x4_t* %B
        %tmp3 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 2
        %tmp6 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 3
	%tmp7 = load <8 x i8>* %C
	%tmp8 = call <8 x i8> @llvm.arm.neon.vtbx4(<8 x i8> %tmp1, <8 x i8> %tmp3, <8 x i8> %tmp4, <8 x i8> %tmp5, <8 x i8> %tmp6, <8 x i8> %tmp7)
	ret <8 x i8> %tmp8
}

declare <8 x i8>  @llvm.arm.neon.vtbl1(<8 x i8>, <8 x i8>) nounwind readnone
declare <8 x i8>  @llvm.arm.neon.vtbl2(<8 x i8>, <8 x i8>, <8 x i8>) nounwind readnone
declare <8 x i8>  @llvm.arm.neon.vtbl3(<8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>) nounwind readnone
declare <8 x i8>  @llvm.arm.neon.vtbl4(<8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vtbx1(<8 x i8>, <8 x i8>, <8 x i8>) nounwind readnone
declare <8 x i8>  @llvm.arm.neon.vtbx2(<8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>) nounwind readnone
declare <8 x i8>  @llvm.arm.neon.vtbx3(<8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>) nounwind readnone
declare <8 x i8>  @llvm.arm.neon.vtbx4(<8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>) nounwind readnone
