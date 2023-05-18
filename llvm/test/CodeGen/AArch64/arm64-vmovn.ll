; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <8 x i8> @xtn8b(<8 x i16> %A) nounwind {
;CHECK-LABEL: xtn8b:
;CHECK-NOT: ld1
;CHECK: xtn.8b v0, v0
;CHECK-NEXT: ret
  %tmp3 = trunc <8 x i16> %A to <8 x i8>
        ret <8 x i8> %tmp3
}

define <4 x i16> @xtn4h(<4 x i32> %A) nounwind {
;CHECK-LABEL: xtn4h:
;CHECK-NOT: ld1
;CHECK: xtn.4h v0, v0
;CHECK-NEXT: ret
  %tmp3 = trunc <4 x i32> %A to <4 x i16>
        ret <4 x i16> %tmp3
}

define <2 x i32> @xtn2s(<2 x i64> %A) nounwind {
;CHECK-LABEL: xtn2s:
;CHECK-NOT: ld1
;CHECK: xtn.2s v0, v0
;CHECK-NEXT: ret
  %tmp3 = trunc <2 x i64> %A to <2 x i32>
        ret <2 x i32> %tmp3
}

define <16 x i8> @xtn2_16b(<8 x i8> %ret, <8 x i16> %A) nounwind {
;CHECK-LABEL: xtn2_16b:
;CHECK-NOT: ld1
;CHECK: xtn2.16b v0, v1
;CHECK-NEXT: ret
        %tmp3 = trunc <8 x i16> %A to <8 x i8>
        %res = shufflevector <8 x i8> %ret, <8 x i8> %tmp3, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        ret <16 x i8> %res
}

define <8 x i16> @xtn2_8h(<4 x i16> %ret, <4 x i32> %A) nounwind {
;CHECK-LABEL: xtn2_8h:
;CHECK-NOT: ld1
;CHECK: xtn2.8h v0, v1
;CHECK-NEXT: ret
        %tmp3 = trunc <4 x i32> %A to <4 x i16>
        %res = shufflevector <4 x i16> %ret, <4 x i16> %tmp3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
        ret <8 x i16> %res
}

define <4 x i32> @xtn2_4s(<2 x i32> %ret, <2 x i64> %A) nounwind {
;CHECK-LABEL: xtn2_4s:
;CHECK-NOT: ld1
;CHECK: xtn2.4s v0, v1
;CHECK-NEXT: ret
        %tmp3 = trunc <2 x i64> %A to <2 x i32>
        %res = shufflevector <2 x i32> %ret, <2 x i32> %tmp3, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
        ret <4 x i32> %res
}

define <8 x i8> @sqxtn8b(<8 x i16> %A) nounwind {
;CHECK-LABEL: sqxtn8b:
;CHECK-NOT: ld1
;CHECK: sqxtn.8b v0, v0
;CHECK-NEXT: ret
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.sqxtn.v8i8(<8 x i16> %A)
        ret <8 x i8> %tmp3
}

define <4 x i16> @sqxtn4h(<4 x i32> %A) nounwind {
;CHECK-LABEL: sqxtn4h:
;CHECK-NOT: ld1
;CHECK: sqxtn.4h v0, v0
;CHECK-NEXT: ret
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.sqxtn.v4i16(<4 x i32> %A)
        ret <4 x i16> %tmp3
}

define <2 x i32> @sqxtn2s(<2 x i64> %A) nounwind {
;CHECK-LABEL: sqxtn2s:
;CHECK-NOT: ld1
;CHECK: sqxtn.2s v0, v0
;CHECK-NEXT: ret
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.sqxtn.v2i32(<2 x i64> %A)
        ret <2 x i32> %tmp3
}

define <16 x i8> @sqxtn2_16b(<8 x i8> %ret, <8 x i16> %A) nounwind {
;CHECK-LABEL: sqxtn2_16b:
;CHECK-NOT: ld1
;CHECK: sqxtn2.16b v0, v1
;CHECK-NEXT: ret
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.sqxtn.v8i8(<8 x i16> %A)
        %res = shufflevector <8 x i8> %ret, <8 x i8> %tmp3, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        ret <16 x i8> %res
}

define <8 x i16> @sqxtn2_8h(<4 x i16> %ret, <4 x i32> %A) nounwind {
;CHECK-LABEL: sqxtn2_8h:
;CHECK-NOT: ld1
;CHECK: sqxtn2.8h v0, v1
;CHECK-NEXT: ret
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.sqxtn.v4i16(<4 x i32> %A)
        %res = shufflevector <4 x i16> %ret, <4 x i16> %tmp3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
        ret <8 x i16> %res
}

define <4 x i32> @sqxtn2_4s(<2 x i32> %ret, <2 x i64> %A) nounwind {
;CHECK-LABEL: sqxtn2_4s:
;CHECK-NOT: ld1
;CHECK: sqxtn2.4s v0, v1
;CHECK-NEXT: ret
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.sqxtn.v2i32(<2 x i64> %A)
        %res = shufflevector <2 x i32> %ret, <2 x i32> %tmp3, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
        ret <4 x i32> %res
}

declare <8 x i8>  @llvm.aarch64.neon.sqxtn.v8i8(<8 x i16>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.sqxtn.v4i16(<4 x i32>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.sqxtn.v2i32(<2 x i64>) nounwind readnone

define <8 x i8> @uqxtn8b(<8 x i16> %A) nounwind {
;CHECK-LABEL: uqxtn8b:
;CHECK-NOT: ld1
;CHECK: uqxtn.8b v0, v0
;CHECK-NEXT: ret
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.uqxtn.v8i8(<8 x i16> %A)
        ret <8 x i8> %tmp3
}

define <4 x i16> @uqxtn4h(<4 x i32> %A) nounwind {
;CHECK-LABEL: uqxtn4h:
;CHECK-NOT: ld1
;CHECK: uqxtn.4h v0, v0
;CHECK-NEXT: ret
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.uqxtn.v4i16(<4 x i32> %A)
        ret <4 x i16> %tmp3
}

define <2 x i32> @uqxtn2s(<2 x i64> %A) nounwind {
;CHECK-LABEL: uqxtn2s:
;CHECK-NOT: ld1
;CHECK: uqxtn.2s v0, v0
;CHECK-NEXT: ret
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.uqxtn.v2i32(<2 x i64> %A)
        ret <2 x i32> %tmp3
}

define <16 x i8> @uqxtn2_16b(<8 x i8> %ret, <8 x i16> %A) nounwind {
;CHECK-LABEL: uqxtn2_16b:
;CHECK-NOT: ld1
;CHECK: uqxtn2.16b v0, v1
;CHECK-NEXT: ret
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.uqxtn.v8i8(<8 x i16> %A)
        %res = shufflevector <8 x i8> %ret, <8 x i8> %tmp3, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        ret <16 x i8> %res
}

define <8 x i16> @uqxtn2_8h(<4 x i16> %ret, <4 x i32> %A) nounwind {
;CHECK-LABEL: uqxtn2_8h:
;CHECK-NOT: ld1
;CHECK: uqxtn2.8h v0, v1
;CHECK-NEXT: ret
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.uqxtn.v4i16(<4 x i32> %A)
        %res = shufflevector <4 x i16> %ret, <4 x i16> %tmp3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
        ret <8 x i16> %res
}

define <4 x i32> @uqxtn2_4s(<2 x i32> %ret, <2 x i64> %A) nounwind {
;CHECK-LABEL: uqxtn2_4s:
;CHECK-NOT: ld1
;CHECK: uqxtn2.4s v0, v1
;CHECK-NEXT: ret
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.uqxtn.v2i32(<2 x i64> %A)
        %res = shufflevector <2 x i32> %ret, <2 x i32> %tmp3, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
        ret <4 x i32> %res
}

declare <8 x i8>  @llvm.aarch64.neon.uqxtn.v8i8(<8 x i16>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.uqxtn.v4i16(<4 x i32>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.uqxtn.v2i32(<2 x i64>) nounwind readnone

define <8 x i8> @sqxtun8b(<8 x i16> %A) nounwind {
;CHECK-LABEL: sqxtun8b:
;CHECK-NOT: ld1
;CHECK: sqxtun.8b v0, v0
;CHECK-NEXT: ret
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.sqxtun.v8i8(<8 x i16> %A)
        ret <8 x i8> %tmp3
}

define <4 x i16> @sqxtun4h(<4 x i32> %A) nounwind {
;CHECK-LABEL: sqxtun4h:
;CHECK-NOT: ld1
;CHECK: sqxtun.4h v0, v0
;CHECK-NEXT: ret
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.sqxtun.v4i16(<4 x i32> %A)
        ret <4 x i16> %tmp3
}

define <2 x i32> @sqxtun2s(<2 x i64> %A) nounwind {
;CHECK-LABEL: sqxtun2s:
;CHECK-NOT: ld1
;CHECK: sqxtun.2s v0, v0
;CHECK-NEXT: ret
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.sqxtun.v2i32(<2 x i64> %A)
        ret <2 x i32> %tmp3
}

define <16 x i8> @sqxtun2_16b(<8 x i8> %ret, <8 x i16> %A) nounwind {
;CHECK-LABEL: sqxtun2_16b:
;CHECK-NOT: ld1
;CHECK: sqxtun2.16b v0, v1
;CHECK-NEXT: ret
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.sqxtun.v8i8(<8 x i16> %A)
        %res = shufflevector <8 x i8> %ret, <8 x i8> %tmp3, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        ret <16 x i8> %res
}

define <8 x i16> @sqxtun2_8h(<4 x i16> %ret, <4 x i32> %A) nounwind {
;CHECK-LABEL: sqxtun2_8h:
;CHECK-NOT: ld1
;CHECK: sqxtun2.8h v0, v1
;CHECK-NEXT: ret
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.sqxtun.v4i16(<4 x i32> %A)
        %res = shufflevector <4 x i16> %ret, <4 x i16> %tmp3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
        ret <8 x i16> %res
}

define <4 x i32> @sqxtun2_4s(<2 x i32> %ret, <2 x i64> %A) nounwind {
;CHECK-LABEL: sqxtun2_4s:
;CHECK-NOT: ld1
;CHECK: sqxtun2.4s v0, v1
;CHECK-NEXT: ret
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.sqxtun.v2i32(<2 x i64> %A)
        %res = shufflevector <2 x i32> %ret, <2 x i32> %tmp3, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
        ret <4 x i32> %res
}

declare <8 x i8>  @llvm.aarch64.neon.sqxtun.v8i8(<8 x i16>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.sqxtun.v4i16(<4 x i32>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.sqxtun.v2i32(<2 x i64>) nounwind readnone

