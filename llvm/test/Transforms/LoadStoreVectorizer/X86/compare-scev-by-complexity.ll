; RUN: opt -load-store-vectorizer %s -S | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='function(load-store-vectorizer)' %s -S | FileCheck %s

; Check that setting wrapping flags after a SCEV node is created
; does not invalidate "sorted by complexity" invariant for
; operands of commutative and associative SCEV operators.

target triple = "x86_64--"

@global_value0 = external constant i32
@global_value1 = external constant i32
@other_value = external global float
@a = external global float
@b = external global float
@c = external global float
@d = external global float
@plus1 = external global i32
@cnd = external global i8

; Function Attrs: nounwind
define void @main() local_unnamed_addr #0 {
; CHECK-LABEL: @main()
; CHECK: [[PTR:%[0-9]+]] = bitcast float* %preheader.load0.address to <2 x float>*
; CHECK:  = load <2 x float>, <2 x float>* [[PTR]]
; CHECK-LABEL: for.body23:
entry:
  %tmp = load i32, i32* @global_value0, !range !0
  %tmp2 = load i32, i32* @global_value1
  %and.i.i = and i32 %tmp2, 2
  %add.nuw.nsw.i.i = add nuw nsw i32 %and.i.i, 0
  %mul.i.i = shl nuw nsw i32 %add.nuw.nsw.i.i, 1
  %and6.i.i = and i32 %tmp2, 3
  %and9.i.i = and i32 %tmp2, 4
  %add.nuw.nsw10.i.i = add nuw nsw i32 %and6.i.i, %and9.i.i
  %conv3.i42.i = add nuw nsw i32 %mul.i.i, 1
  %reass.add346.7 = add nuw nsw i32 %add.nuw.nsw10.i.i, 56
  %reass.mul347.7 = mul nuw nsw i32 %tmp, %reass.add346.7
  %add7.i.7 = add nuw nsw i32 %reass.mul347.7, 0
  %preheader.address0.idx = add nuw nsw i32 %add7.i.7, %mul.i.i
  %preheader.address0.idx.zext = zext i32 %preheader.address0.idx to i64
  %preheader.load0.address = getelementptr inbounds float, float* @other_value, i64 %preheader.address0.idx.zext
  %preheader.load0. = load float, float* %preheader.load0.address, align 4, !tbaa !1
  %common.address.idx = add nuw nsw i32 %add7.i.7, %conv3.i42.i
  %preheader.header.common.address.idx.zext = zext i32 %common.address.idx to i64
  %preheader.load1.address = getelementptr inbounds float, float* @other_value, i64 %preheader.header.common.address.idx.zext
  %preheader.load1. = load float, float* %preheader.load1.address, align 4, !tbaa !1
  br label %for.body23

for.body23:                                       ; preds = %for.body23, %entry
  %loop.header.load0.address = getelementptr inbounds float, float* @other_value, i64 %preheader.header.common.address.idx.zext
  %loop.header.load0. = load float, float* %loop.header.load0.address, align 4, !tbaa !1
  %reass.mul343.7 = mul nuw nsw i32 %reass.add346.7, 72
  %add7.i286.7.7 = add nuw nsw i32 %reass.mul343.7, 56
  %add9.i288.7.7 = add nuw nsw i32 %add7.i286.7.7, %mul.i.i
  %loop.header.address1.idx = add nuw nsw i32 %add9.i288.7.7, 1
  %loop.header.address1.idx.zext = zext i32 %loop.header.address1.idx to i64
  %loop.header.load1.address = getelementptr inbounds float, float* @other_value, i64 %loop.header.address1.idx.zext
  %loop.header.load1. = load float, float* %loop.header.load1.address, align 4, !tbaa !1
  store float %preheader.load0., float* @a, align 4, !tbaa !1
  store float %preheader.load1., float* @b, align 4, !tbaa !1
  store float %loop.header.load0., float* @c, align 4, !tbaa !1
  store float %loop.header.load1., float* @d, align 4, !tbaa !1
  %loaded.cnd = load i8, i8* @cnd
  %condition = trunc i8 %loaded.cnd to i1
  br i1 %condition, label %for.body23, label %exit

exit:
  ret void
}

attributes #0 = { nounwind }

!0 = !{i32 0, i32 65536}
!1 = !{!2, !2, i64 0}
!2 = !{!"float", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}
