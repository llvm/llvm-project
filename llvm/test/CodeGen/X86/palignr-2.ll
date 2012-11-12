; RUN: llc < %s -march=x86 -mattr=+ssse3 | FileCheck %s
; rdar://7341330

@a = global [4 x i32] [i32 4, i32 5, i32 6, i32 7], align 16 ; <[4 x i32]*> [#uses=1]
@c = common global [4 x i32] zeroinitializer, align 16 ; <[4 x i32]*> [#uses=1]
@b = global [4 x i32] [i32 0, i32 1, i32 2, i32 3], align 16 ; <[4 x i32]*> [#uses=1]

define void @t1(<2 x i64> %a, <2 x i64> %b) nounwind ssp {
entry:
; CHECK: t1:
; palignr $3, %xmm1, %xmm0
  %0 = tail call <2 x i64> @llvm.x86.ssse3.palign.r.128(<2 x i64> %a, <2 x i64> %b, i8 24) nounwind readnone
  store <2 x i64> %0, <2 x i64>* bitcast ([4 x i32]* @c to <2 x i64>*), align 16
  ret void
}

declare <2 x i64> @llvm.x86.ssse3.palign.r.128(<2 x i64>, <2 x i64>, i8) nounwind readnone

define void @t2() nounwind ssp {
entry:
; CHECK: t2:
; palignr $4, _b, %xmm0
  %0 = load <2 x i64>* bitcast ([4 x i32]* @b to <2 x i64>*), align 16 ; <<2 x i64>> [#uses=1]
  %1 = load <2 x i64>* bitcast ([4 x i32]* @a to <2 x i64>*), align 16 ; <<2 x i64>> [#uses=1]
  %2 = tail call <2 x i64> @llvm.x86.ssse3.palign.r.128(<2 x i64> %1, <2 x i64> %0, i8 32) nounwind readnone
  store <2 x i64> %2, <2 x i64>* bitcast ([4 x i32]* @c to <2 x i64>*), align 16
  ret void
}
