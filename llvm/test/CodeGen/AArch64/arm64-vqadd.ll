; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <8 x i8> @sqadd8b(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: sqadd8b:
;CHECK: sqadd.8b
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <8 x i8> @llvm.aarch64.neon.sqadd.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @sqadd4h(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: sqadd4h:
;CHECK: sqadd.4h
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <4 x i16> @llvm.aarch64.neon.sqadd.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @sqadd2s(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: sqadd2s:
;CHECK: sqadd.2s
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.sqadd.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <8 x i8> @uqadd8b(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: uqadd8b:
;CHECK: uqadd.8b
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <8 x i8> @llvm.aarch64.neon.uqadd.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @uqadd4h(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: uqadd4h:
;CHECK: uqadd.4h
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <4 x i16> @llvm.aarch64.neon.uqadd.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @uqadd2s(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: uqadd2s:
;CHECK: uqadd.2s
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.uqadd.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <16 x i8> @sqadd16b(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: sqadd16b:
;CHECK: sqadd.16b
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = call <16 x i8> @llvm.aarch64.neon.sqadd.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @sqadd8h(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: sqadd8h:
;CHECK: sqadd.8h
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = call <8 x i16> @llvm.aarch64.neon.sqadd.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @sqadd4s(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: sqadd4s:
;CHECK: sqadd.4s
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @sqadd2d(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: sqadd2d:
;CHECK: sqadd.2d
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = load <2 x i64>, ptr %B
	%tmp3 = call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

define <16 x i8> @uqadd16b(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: uqadd16b:
;CHECK: uqadd.16b
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = call <16 x i8> @llvm.aarch64.neon.uqadd.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @uqadd8h(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: uqadd8h:
;CHECK: uqadd.8h
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = call <8 x i16> @llvm.aarch64.neon.uqadd.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @uqadd4s(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: uqadd4s:
;CHECK: uqadd.4s
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.uqadd.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @uqadd2d(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: uqadd2d:
;CHECK: uqadd.2d
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = load <2 x i64>, ptr %B
	%tmp3 = call <2 x i64> @llvm.aarch64.neon.uqadd.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

declare <8 x i8>  @llvm.aarch64.neon.sqadd.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.sqadd.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.sqadd.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.aarch64.neon.sqadd.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <8 x i8>  @llvm.aarch64.neon.uqadd.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.uqadd.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.uqadd.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.aarch64.neon.uqadd.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <16 x i8> @llvm.aarch64.neon.sqadd.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.sqadd.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

declare <16 x i8> @llvm.aarch64.neon.uqadd.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.uqadd.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.uqadd.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.uqadd.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i8> @usqadd8b(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: usqadd8b:
;CHECK: usqadd.8b
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <8 x i8> @llvm.aarch64.neon.usqadd.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @usqadd4h(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: usqadd4h:
;CHECK: usqadd.4h
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <4 x i16> @llvm.aarch64.neon.usqadd.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @usqadd2s(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: usqadd2s:
;CHECK: usqadd.2s
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.usqadd.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <16 x i8> @usqadd16b(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: usqadd16b:
;CHECK: usqadd.16b
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = call <16 x i8> @llvm.aarch64.neon.usqadd.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @usqadd8h(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: usqadd8h:
;CHECK: usqadd.8h
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = call <8 x i16> @llvm.aarch64.neon.usqadd.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @usqadd4s(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: usqadd4s:
;CHECK: usqadd.4s
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.usqadd.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @usqadd2d(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: usqadd2d:
;CHECK: usqadd.2d
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = load <2 x i64>, ptr %B
	%tmp3 = call <2 x i64> @llvm.aarch64.neon.usqadd.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

define i64 @usqadd_d(i64 %l, i64 %r) nounwind {
; CHECK-LABEL: usqadd_d:
; CHECK: usqadd {{d[0-9]+}}, {{d[0-9]+}}
  %sum = call i64 @llvm.aarch64.neon.usqadd.i64(i64 %l, i64 %r)
  ret i64 %sum
}

define i32 @usqadd_s(i32 %l, i32 %r) nounwind {
; CHECK-LABEL: usqadd_s:
; CHECK: usqadd {{s[0-9]+}}, {{s[0-9]+}}
  %sum = call i32 @llvm.aarch64.neon.usqadd.i32(i32 %l, i32 %r)
  ret i32 %sum
}

declare <8 x i8>  @llvm.aarch64.neon.usqadd.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.usqadd.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.usqadd.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.aarch64.neon.usqadd.v1i64(<1 x i64>, <1 x i64>) nounwind readnone
declare i64 @llvm.aarch64.neon.usqadd.i64(i64, i64) nounwind readnone
declare i32 @llvm.aarch64.neon.usqadd.i32(i32, i32) nounwind readnone

declare <16 x i8> @llvm.aarch64.neon.usqadd.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.usqadd.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.usqadd.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.usqadd.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i8> @suqadd8b(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: suqadd8b:
;CHECK: suqadd.8b
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <8 x i8> @llvm.aarch64.neon.suqadd.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @suqadd4h(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: suqadd4h:
;CHECK: suqadd.4h
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <4 x i16> @llvm.aarch64.neon.suqadd.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @suqadd2s(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: suqadd2s:
;CHECK: suqadd.2s
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.suqadd.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <16 x i8> @suqadd16b(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: suqadd16b:
;CHECK: suqadd.16b
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = call <16 x i8> @llvm.aarch64.neon.suqadd.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @suqadd8h(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: suqadd8h:
;CHECK: suqadd.8h
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = call <8 x i16> @llvm.aarch64.neon.suqadd.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @suqadd4s(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: suqadd4s:
;CHECK: suqadd.4s
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.suqadd.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @suqadd2d(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: suqadd2d:
;CHECK: suqadd.2d
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = load <2 x i64>, ptr %B
	%tmp3 = call <2 x i64> @llvm.aarch64.neon.suqadd.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

define <1 x i64> @suqadd_1d(<1 x i64> %l, <1 x i64> %r) nounwind {
; CHECK-LABEL: suqadd_1d:
; CHECK: suqadd {{d[0-9]+}}, {{d[0-9]+}}
  %sum = call <1 x i64> @llvm.aarch64.neon.suqadd.v1i64(<1 x i64> %l, <1 x i64> %r)
  ret <1 x i64> %sum
}

define i64 @suqadd_d(i64 %l, i64 %r) nounwind {
; CHECK-LABEL: suqadd_d:
; CHECK: suqadd {{d[0-9]+}}, {{d[0-9]+}}
  %sum = call i64 @llvm.aarch64.neon.suqadd.i64(i64 %l, i64 %r)
  ret i64 %sum
}

define i32 @suqadd_s(i32 %l, i32 %r) nounwind {
; CHECK-LABEL: suqadd_s:
; CHECK: suqadd {{s[0-9]+}}, {{s[0-9]+}}
  %sum = call i32 @llvm.aarch64.neon.suqadd.i32(i32 %l, i32 %r)
  ret i32 %sum
}

declare <8 x i8>  @llvm.aarch64.neon.suqadd.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.suqadd.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.suqadd.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.aarch64.neon.suqadd.v1i64(<1 x i64>, <1 x i64>) nounwind readnone
declare i64 @llvm.aarch64.neon.suqadd.i64(i64, i64) nounwind readnone
declare i32 @llvm.aarch64.neon.suqadd.i32(i32, i32) nounwind readnone

declare <16 x i8> @llvm.aarch64.neon.suqadd.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.suqadd.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.suqadd.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.suqadd.v2i64(<2 x i64>, <2 x i64>) nounwind readnone
