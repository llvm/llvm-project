; RUN: opt %s -passes='function(scalarizer,dce)' -scalarize-load-store -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare <4 x float> @ext(<4 x float>)
@g = global <4 x float> zeroinitializer

define void @f1(<4 x float> %init, <4 x float> *%base, i32 %count) {
; CHECK-LABEL: @f1(
; CHECK: entry:
; CHECK:   %init.i0 = extractelement <4 x float> %init, i32 0
; CHECK:   %init.i1 = extractelement <4 x float> %init, i32 1
; CHECK:   %init.i2 = extractelement <4 x float> %init, i32 2
; CHECK:   %init.i3 = extractelement <4 x float> %init, i32 3
; CHECK:   br label %loop
; CHECK: loop:
; CHECK:   %i = phi i32 [ %count, %entry ], [ %nexti, %loop ]
; CHECK:   %acc.i0 = phi float [ %init.i0, %entry ], [ %sel.i0, %loop ]
; CHECK:   %acc.i1 = phi float [ %init.i1, %entry ], [ %sel.i1, %loop ]
; CHECK:   %acc.i2 = phi float [ %init.i2, %entry ], [ %sel.i2, %loop ]
; CHECK:   %acc.i3 = phi float [ %init.i3, %entry ], [ %sel.i3, %loop ]
; CHECK:   %nexti = sub i32 %i, 1
; CHECK:   %ptr = getelementptr <4 x float>, <4 x float>* %base, i32 %i
; CHECK:   %ptr.i0 = bitcast <4 x float>* %ptr to float*
; CHECK:   %val.i0 = load float, float* %ptr.i0, align 16
; CHECK:   %ptr.i1 = getelementptr float, float* %ptr.i0, i32 1
; CHECK:   %val.i1 = load float, float* %ptr.i1, align 4
; CHECK:   %ptr.i2 = getelementptr float, float* %ptr.i0, i32 2
; CHECK:   %val.i2 = load float, float* %ptr.i2, align 8
; CHECK:   %ptr.i3 = getelementptr float, float* %ptr.i0, i32 3
; CHECK:   %val.i3 = load float, float* %ptr.i3, align 4
; CHECK:   %add.i0 = fadd float %val.i0, %val.i2
; CHECK:   %add.i1 = fadd float %val.i1, %val.i3
; CHECK:   %add.i2 = fadd float %acc.i0, %acc.i2
; CHECK:   %add.i3 = fadd float %acc.i1, %acc.i3
; CHECK:   %add.upto0 = insertelement <4 x float> poison, float %add.i0, i32 0
; CHECK:   %add.upto1 = insertelement <4 x float> %add.upto0, float %add.i1, i32 1
; CHECK:   %add.upto2 = insertelement <4 x float> %add.upto1, float %add.i2, i32 2
; CHECK:   %add = insertelement <4 x float> %add.upto2, float %add.i3, i32 3
; CHECK:   %call = call <4 x float> @ext(<4 x float> %add)
; CHECK:   %call.i0 = extractelement <4 x float> %call, i32 0
; CHECK:   %cmp.i0 = fcmp ogt float %call.i0, 1.0
; CHECK:   %call.i1 = extractelement <4 x float> %call, i32 1
; CHECK:   %cmp.i1 = fcmp ogt float %call.i1, 2.0
; CHECK:   %call.i2 = extractelement <4 x float> %call, i32 2
; CHECK:   %cmp.i2 = fcmp ogt float %call.i2, 3.0
; CHECK:   %call.i3 = extractelement <4 x float> %call, i32 3
; CHECK:   %cmp.i3 = fcmp ogt float %call.i3, 4.0
; CHECK:   %sel.i0 = select i1 %cmp.i0, float %call.i0, float 5.0
; CHECK:   %sel.i1 = select i1 %cmp.i1, float %call.i1, float 6.0
; CHECK:   %sel.i2 = select i1 %cmp.i2, float %call.i2, float 7.0
; CHECK:   %sel.i3 = select i1 %cmp.i3, float %call.i3, float 8.0
; CHECK:   store float %sel.i0, float* %ptr.i0
; CHECK:   store float %sel.i1, float* %ptr.i1
; CHECK:   store float %sel.i2, float* %ptr.i2
; CHECK:   store float %sel.i3, float* %ptr.i3
; CHECK:   %test = icmp eq i32 %nexti, 0
; CHECK:   br i1 %test, label %loop, label %exit
; CHECK: exit:
; CHECK:   ret void
entry:
  br label %loop

loop:
  %i = phi i32 [ %count, %entry ], [ %nexti, %loop ]
  %acc = phi <4 x float> [ %init, %entry ], [ %sel, %loop ]
  %nexti = sub i32 %i, 1

  %ptr = getelementptr <4 x float>, <4 x float> *%base, i32 %i
  %val = load <4 x float> , <4 x float> *%ptr
  %dval = bitcast <4 x float> %val to <2 x double>
  %dacc = bitcast <4 x float> %acc to <2 x double>
  %shuffle1 = shufflevector <2 x double> %dval, <2 x double> %dacc,
                            <2 x i32> <i32 0, i32 2>
  %shuffle2 = shufflevector <2 x double> %dval, <2 x double> %dacc,
                            <2 x i32> <i32 1, i32 3>
  %f1 = bitcast <2 x double> %shuffle1 to <4 x float>
  %f2 = bitcast <2 x double> %shuffle2 to <4 x float>
  %add = fadd <4 x float> %f1, %f2
  %call = call <4 x float> @ext(<4 x float> %add)
  %cmp = fcmp ogt <4 x float> %call,
                  <float 1.0, float 2.0, float 3.0, float 4.0>
  %sel = select <4 x i1> %cmp, <4 x float> %call,
                <4 x float> <float 5.0, float 6.0, float 7.0, float 8.0>
  store <4 x float> %sel, <4 x float> *%ptr

  %test = icmp eq i32 %nexti, 0
  br i1 %test, label %loop, label %exit

exit:
  ret void
}

define void @f2(<4 x i32> %init, <4 x i8> *%base, i32 %count) {
; CHECK-LABEL: define void @f2(<4 x i32> %init, <4 x i8>* %base, i32 %count) {
; CHECK: entry:
; CHECK:   %init.i0 = extractelement <4 x i32> %init, i32 0
; CHECK:   %init.i1 = extractelement <4 x i32> %init, i32 1
; CHECK:   %init.i2 = extractelement <4 x i32> %init, i32 2
; CHECK:   %init.i3 = extractelement <4 x i32> %init, i32 3
; CHECK:   br label %loop
; CHECK: loop:
; CHECK:   %i = phi i32 [ %count, %entry ], [ %nexti, %loop ]
; CHECK:   %acc.i0 = phi i32 [ %init.i0, %entry ], [ %sel.i0, %loop ]
; CHECK:   %acc.i1 = phi i32 [ %init.i1, %entry ], [ %sel.i1, %loop ]
; CHECK:   %acc.i2 = phi i32 [ %init.i2, %entry ], [ %sel.i2, %loop ]
; CHECK:   %acc.i3 = phi i32 [ %init.i3, %entry ], [ %sel.i3, %loop ]
; CHECK:   %nexti = sub i32 %i, 1
; CHECK:   %ptr = getelementptr <4 x i8>, <4 x i8>* %base, i32 %i
; CHECK:   %ptr.i0 = bitcast <4 x i8>* %ptr to i8*
; CHECK:   %val.i0 = load i8, i8* %ptr.i0, align 4
; CHECK:   %ptr.i1 = getelementptr i8, i8* %ptr.i0, i32 1
; CHECK:   %val.i1 = load i8, i8* %ptr.i1, align 1
; CHECK:   %ptr.i2 = getelementptr i8, i8* %ptr.i0, i32 2
; CHECK:   %val.i2 = load i8, i8* %ptr.i2, align 2
; CHECK:   %ptr.i3 = getelementptr i8, i8* %ptr.i0, i32 3
; CHECK:   %val.i3 = load i8, i8* %ptr.i3, align 1
; CHECK:   %ext.i0 = sext i8 %val.i0 to i32
; CHECK:   %ext.i1 = sext i8 %val.i1 to i32
; CHECK:   %ext.i2 = sext i8 %val.i2 to i32
; CHECK:   %ext.i3 = sext i8 %val.i3 to i32
; CHECK:   %add.i0 = add i32 %ext.i0, %acc.i0
; CHECK:   %add.i1 = add i32 %ext.i1, %acc.i1
; CHECK:   %add.i2 = add i32 %ext.i2, %acc.i2
; CHECK:   %add.i3 = add i32 %ext.i3, %acc.i3
; CHECK:   %cmp.i0 = icmp slt i32 %add.i0, -10
; CHECK:   %cmp.i1 = icmp slt i32 %add.i1, -11
; CHECK:   %cmp.i2 = icmp slt i32 %add.i2, -12
; CHECK:   %cmp.i3 = icmp slt i32 %add.i3, -13
; CHECK:   %sel.i0 = select i1 %cmp.i0, i32 %add.i0, i32 %i
; CHECK:   %sel.i1 = select i1 %cmp.i1, i32 %add.i1, i32 %i
; CHECK:   %sel.i2 = select i1 %cmp.i2, i32 %add.i2, i32 %i
; CHECK:   %sel.i3 = select i1 %cmp.i3, i32 %add.i3, i32 %i
; CHECK:   %trunc.i0 = trunc i32 %sel.i0 to i8
; CHECK:   %trunc.i1 = trunc i32 %sel.i1 to i8
; CHECK:   %trunc.i2 = trunc i32 %sel.i2 to i8
; CHECK:   %trunc.i3 = trunc i32 %sel.i3 to i8
; CHECK:   store i8 %trunc.i0, i8* %ptr.i0, align 4
; CHECK:   store i8 %trunc.i1, i8* %ptr.i1, align 1
; CHECK:   store i8 %trunc.i2, i8* %ptr.i2, align 2
; CHECK:   store i8 %trunc.i3, i8* %ptr.i3, align 1
; CHECK:   %test = icmp eq i32 %nexti, 0
; CHECK:   br i1 %test, label %loop, label %exit
; CHECK: exit:
; CHECK:   ret void
entry:
  br label %loop

loop:
  %i = phi i32 [ %count, %entry ], [ %nexti, %loop ]
  %acc = phi <4 x i32> [ %init, %entry ], [ %sel, %loop ]
  %nexti = sub i32 %i, 1

  %ptr = getelementptr <4 x i8>, <4 x i8> *%base, i32 %i
  %val = load <4 x i8> , <4 x i8> *%ptr
  %ext = sext <4 x i8> %val to <4 x i32>
  %add = add <4 x i32> %ext, %acc
  %cmp = icmp slt <4 x i32> %add, <i32 -10, i32 -11, i32 -12, i32 -13>
  %single = insertelement <4 x i32> undef, i32 %i, i32 0
  %limit = shufflevector <4 x i32> %single, <4 x i32> undef,
                         <4 x i32> zeroinitializer
  %sel = select <4 x i1> %cmp, <4 x i32> %add, <4 x i32> %limit
  %trunc = trunc <4 x i32> %sel to <4 x i8>
  store <4 x i8> %trunc, <4 x i8> *%ptr

  %test = icmp eq i32 %nexti, 0
  br i1 %test, label %loop, label %exit

exit:
  ret void
}

; Check that !tbaa information is preserved.
define void @f3(<4 x i32> *%src, <4 x i32> *%dst) {
; CHECK-LABEL: @f3(
; CHECK: %val.i0 = load i32, i32* %src.i0, align 16, !tbaa ![[TAG:[0-9]*]]
; CHECK: %val.i1 = load i32, i32* %src.i1, align 4, !tbaa ![[TAG]]
; CHECK: %val.i2 = load i32, i32* %src.i2, align 8, !tbaa ![[TAG]]
; CHECK: %val.i3 = load i32, i32* %src.i3, align 4, !tbaa ![[TAG]]
; CHECK: store i32 %add.i0, i32* %dst.i0, align 16, !tbaa ![[TAG:[0-9]*]]
; CHECK: store i32 %add.i1, i32* %dst.i1, align 4, !tbaa ![[TAG]]
; CHECK: store i32 %add.i2, i32* %dst.i2, align 8, !tbaa ![[TAG]]
; CHECK: store i32 %add.i3, i32* %dst.i3, align 4, !tbaa ![[TAG]]
; CHECK: ret void
  %val = load <4 x i32> , <4 x i32> *%src, !tbaa !1
  %add = add <4 x i32> %val, %val
  store <4 x i32> %add, <4 x i32> *%dst, !tbaa !2
  ret void
}

; Check that !tbaa.struct information is preserved.
define void @f4(<4 x i32> *%src, <4 x i32> *%dst) {
; CHECK-LABEL: @f4(
; CHECK: %val.i0 = load i32, i32* %src.i0, align 16, !tbaa.struct ![[TAG:[0-9]*]]
; CHECK: %val.i1 = load i32, i32* %src.i1, align 4, !tbaa.struct ![[TAG]]
; CHECK: %val.i2 = load i32, i32* %src.i2, align 8, !tbaa.struct ![[TAG]]
; CHECK: %val.i3 = load i32, i32* %src.i3, align 4, !tbaa.struct ![[TAG]]
; CHECK: store i32 %add.i0, i32* %dst.i0, align 16, !tbaa.struct ![[TAG]]
; CHECK: store i32 %add.i1, i32* %dst.i1, align 4, !tbaa.struct ![[TAG]]
; CHECK: store i32 %add.i2, i32* %dst.i2, align 8, !tbaa.struct ![[TAG]]
; CHECK: store i32 %add.i3, i32* %dst.i3, align 4, !tbaa.struct ![[TAG]]
; CHECK: ret void
  %val = load <4 x i32> , <4 x i32> *%src, !tbaa.struct !5
  %add = add <4 x i32> %val, %val
  store <4 x i32> %add, <4 x i32> *%dst, !tbaa.struct !5
  ret void
}

; Check that llvm.access.group information is preserved.
define void @f5(i32 %count, <4 x i32> *%src, <4 x i32> *%dst) {
; CHECK-LABEL: @f5(
; CHECK: %val.i0 = load i32, i32* %this_src.i0, align 16, !llvm.access.group ![[TAG:[0-9]*]]
; CHECK: %val.i1 = load i32, i32* %this_src.i1, align 4, !llvm.access.group ![[TAG]]
; CHECK: %val.i2 = load i32, i32* %this_src.i2, align 8, !llvm.access.group ![[TAG]]
; CHECK: %val.i3 = load i32, i32* %this_src.i3, align 4, !llvm.access.group ![[TAG]]
; CHECK: store i32 %add.i0, i32* %this_dst.i0, align 16, !llvm.access.group ![[TAG]]
; CHECK: store i32 %add.i1, i32* %this_dst.i1, align 4, !llvm.access.group ![[TAG]]
; CHECK: store i32 %add.i2, i32* %this_dst.i2, align 8, !llvm.access.group ![[TAG]]
; CHECK: store i32 %add.i3, i32* %this_dst.i3, align 4, !llvm.access.group ![[TAG]]
; CHECK: ret void
entry:
  br label %loop

loop:
  %index = phi i32 [ 0, %entry ], [ %next_index, %loop ]
  %this_src = getelementptr <4 x i32>, <4 x i32> *%src, i32 %index
  %this_dst = getelementptr <4 x i32>, <4 x i32> *%dst, i32 %index
  %val = load <4 x i32> , <4 x i32> *%this_src, !llvm.access.group !13
  %add = add <4 x i32> %val, %val
  store <4 x i32> %add, <4 x i32> *%this_dst, !llvm.access.group !13
  %next_index = add i32 %index, -1
  %continue = icmp ne i32 %next_index, %count
  br i1 %continue, label %loop, label %end, !llvm.loop !3

end:
  ret void
}

; Check that fpmath information is preserved.
define <4 x float> @f6(<4 x float> %x) {
; CHECK-LABEL: @f6(
; CHECK: %x.i0 = extractelement <4 x float> %x, i32 0
; CHECK: %res.i0 = fadd float %x.i0, 1.0{{[e+0]*}}, !fpmath ![[TAG:[0-9]*]]
; CHECK: %x.i1 = extractelement <4 x float> %x, i32 1
; CHECK: %res.i1 = fadd float %x.i1, 2.0{{[e+0]*}}, !fpmath ![[TAG]]
; CHECK: %x.i2 = extractelement <4 x float> %x, i32 2
; CHECK: %res.i2 = fadd float %x.i2, 3.0{{[e+0]*}}, !fpmath ![[TAG]]
; CHECK: %x.i3 = extractelement <4 x float> %x, i32 3
; CHECK: %res.i3 = fadd float %x.i3, 4.0{{[e+0]*}}, !fpmath ![[TAG]]
; CHECK: %res.upto0 = insertelement <4 x float> poison, float %res.i0, i32 0
; CHECK: %res.upto1 = insertelement <4 x float> %res.upto0, float %res.i1, i32 1
; CHECK: %res.upto2 = insertelement <4 x float> %res.upto1, float %res.i2, i32 2
; CHECK: %res = insertelement <4 x float> %res.upto2, float %res.i3, i32 3
; CHECK: ret <4 x float> %res
  %res = fadd <4 x float> %x, <float 1.0, float 2.0, float 3.0, float 4.0>,
    !fpmath !4
  ret <4 x float> %res
}

; Check that random metadata isn't kept.
define void @f7(<4 x i32> *%src, <4 x i32> *%dst) {
; CHECK-LABEL: @f7(
; CHECK-NOT: !foo
; CHECK: ret void
  %val = load <4 x i32> , <4 x i32> *%src, !foo !5
  %add = add <4 x i32> %val, %val
  store <4 x i32> %add, <4 x i32> *%dst, !foo !5
  ret void
}

; Test GEP with vectors.
define void @f8(<4 x float *> *%dest, <4 x float *> %ptr0, <4 x i32> %i0,
                float *%other) {
; CHECK-LABEL: @f8(
; CHECK: %dest.i0 = bitcast <4 x float*>* %dest to float**
; CHECK: %dest.i1 = getelementptr float*, float** %dest.i0, i32 1
; CHECK: %dest.i2 = getelementptr float*, float** %dest.i0, i32 2
; CHECK: %dest.i3 = getelementptr float*, float** %dest.i0, i32 3
; CHECK: %ptr0.i0 = extractelement <4 x float*> %ptr0, i32 0
; CHECK: %ptr0.i2 = extractelement <4 x float*> %ptr0, i32 2
; CHECK: %ptr0.i3 = extractelement <4 x float*> %ptr0, i32 3
; CHECK: %i0.i1 = extractelement <4 x i32> %i0, i32 1
; CHECK: %i0.i3 = extractelement <4 x i32> %i0, i32 3
; CHECK: %val.i0 = getelementptr float, float* %ptr0.i0, i32 100
; CHECK: %val.i1 = getelementptr float, float* %other, i32 %i0.i1
; CHECK: %val.i2 = getelementptr float, float* %ptr0.i2, i32 100
; CHECK: %val.i3 = getelementptr float, float* %ptr0.i3, i32 %i0.i3
; CHECK: store float* %val.i0, float** %dest.i0, align 32
; CHECK: store float* %val.i1, float** %dest.i1, align 8
; CHECK: store float* %val.i2, float** %dest.i2, align 16
; CHECK: store float* %val.i3, float** %dest.i3, align 8
; CHECK: ret void
  %i1 = insertelement <4 x i32> %i0, i32 100, i32 0
  %i2 = insertelement <4 x i32> %i1, i32 100, i32 2
  %ptr1 = insertelement <4 x float *> %ptr0, float *%other, i32 1
  %val = getelementptr float, <4 x float *> %ptr1, <4 x i32> %i2
  store <4 x float *> %val, <4 x float *> *%dest
  ret void
}

; Test the handling of unaligned loads.
define void @f9(<4 x float> *%dest, <4 x float> *%src) {
; CHECK: @f9(
; CHECK: %dest.i0 = bitcast <4 x float>* %dest to float*
; CHECK: %dest.i1 = getelementptr float, float* %dest.i0, i32 1
; CHECK: %dest.i2 = getelementptr float, float* %dest.i0, i32 2
; CHECK: %dest.i3 = getelementptr float, float* %dest.i0, i32 3
; CHECK: %src.i0 = bitcast <4 x float>* %src to float*
; CHECK: %val.i0 = load float, float* %src.i0, align 4
; CHECK: %src.i1 = getelementptr float, float* %src.i0, i32 1
; CHECK: %val.i1 = load float, float* %src.i1, align 4
; CHECK: %src.i2 = getelementptr float, float* %src.i0, i32 2
; CHECK: %val.i2 = load float, float* %src.i2, align 4
; CHECK: %src.i3 = getelementptr float, float* %src.i0, i32 3
; CHECK: %val.i3 = load float, float* %src.i3, align 4
; CHECK: store float %val.i0, float* %dest.i0, align 8
; CHECK: store float %val.i1, float* %dest.i1, align 4
; CHECK: store float %val.i2, float* %dest.i2, align 8
; CHECK: store float %val.i3, float* %dest.i3, align 4
; CHECK: ret void
  %val = load <4 x float> , <4 x float> *%src, align 4
  store <4 x float> %val, <4 x float> *%dest, align 8
  ret void
}

; ...and again with subelement alignment.
define void @f10(<4 x float> *%dest, <4 x float> *%src) {
; CHECK: @f10(
; CHECK: %dest.i0 = bitcast <4 x float>* %dest to float*
; CHECK: %dest.i1 = getelementptr float, float* %dest.i0, i32 1
; CHECK: %dest.i2 = getelementptr float, float* %dest.i0, i32 2
; CHECK: %dest.i3 = getelementptr float, float* %dest.i0, i32 3
; CHECK: %src.i0 = bitcast <4 x float>* %src to float*
; CHECK: %val.i0 = load float, float* %src.i0, align 1
; CHECK: %src.i1 = getelementptr float, float* %src.i0, i32 1
; CHECK: %val.i1 = load float, float* %src.i1, align 1
; CHECK: %src.i2 = getelementptr float, float* %src.i0, i32 2
; CHECK: %val.i2 = load float, float* %src.i2, align 1
; CHECK: %src.i3 = getelementptr float, float* %src.i0, i32 3
; CHECK: %val.i3 = load float, float* %src.i3, align 1
; CHECK: store float %val.i0, float* %dest.i0, align 2
; CHECK: store float %val.i1, float* %dest.i1, align 2
; CHECK: store float %val.i2, float* %dest.i2, align 2
; CHECK: store float %val.i3, float* %dest.i3, align 2
; CHECK: ret void
  %val = load <4 x float> , <4 x float> *%src, align 1
  store <4 x float> %val, <4 x float> *%dest, align 2
  ret void
}

; Test that sub-byte loads aren't scalarized.
define void @f11(<32 x i1> *%dest, <32 x i1> *%src0) {
; CHECK: @f11(
; CHECK: %val0 = load <32 x i1>, <32 x i1>* %src0
; CHECK: %val1 = load <32 x i1>, <32 x i1>* %src1
; CHECK: store <32 x i1> %and, <32 x i1>* %dest
; CHECK: ret void
  %src1 = getelementptr <32 x i1>, <32 x i1> *%src0, i32 1
  %val0 = load <32 x i1> , <32 x i1> *%src0
  %val1 = load <32 x i1> , <32 x i1> *%src1
  %and = and <32 x i1> %val0, %val1
  store <32 x i1> %and, <32 x i1> *%dest
  ret void
}

; Test vector GEPs with more than one index.
define void @f13(<4 x float *> *%dest, <4 x [4 x float] *> %ptr, <4 x i32> %i,
                 float *%other) {
; CHECK-LABEL: @f13(
; CHECK: %dest.i0 = bitcast <4 x float*>* %dest to float**
; CHECK: %dest.i1 = getelementptr float*, float** %dest.i0, i32 1
; CHECK: %dest.i2 = getelementptr float*, float** %dest.i0, i32 2
; CHECK: %dest.i3 = getelementptr float*, float** %dest.i0, i32 3
; CHECK: %i.i0 = extractelement <4 x i32> %i, i32 0
; CHECK: %ptr.i0 = extractelement <4 x [4 x float]*> %ptr, i32 0
; CHECK: %val.i0 = getelementptr inbounds [4 x float], [4 x float]* %ptr.i0, i32 0, i32 %i.i0
; CHECK: %i.i1 = extractelement <4 x i32> %i, i32 1
; CHECK: %ptr.i1 = extractelement <4 x [4 x float]*> %ptr, i32 1
; CHECK: %val.i1 = getelementptr inbounds [4 x float], [4 x float]* %ptr.i1, i32 1, i32 %i.i1
; CHECK: %i.i2 = extractelement <4 x i32> %i, i32 2
; CHECK: %ptr.i2 = extractelement <4 x [4 x float]*> %ptr, i32 2
; CHECK: %val.i2 = getelementptr inbounds [4 x float], [4 x float]* %ptr.i2, i32 2, i32 %i.i2
; CHECK: %i.i3 = extractelement <4 x i32> %i, i32 3
; CHECK: %ptr.i3 = extractelement <4 x [4 x float]*> %ptr, i32 3
; CHECK: %val.i3 = getelementptr inbounds [4 x float], [4 x float]* %ptr.i3, i32 3, i32 %i.i3
; CHECK: store float* %val.i0, float** %dest.i0, align 32
; CHECK: store float* %val.i1, float** %dest.i1, align 8
; CHECK: store float* %val.i2, float** %dest.i2, align 16
; CHECK: store float* %val.i3, float** %dest.i3, align 8
; CHECK: ret void
  %val = getelementptr inbounds [4 x float], <4 x [4 x float] *> %ptr,
                                <4 x i32> <i32 0, i32 1, i32 2, i32 3>,
                                <4 x i32> %i
  store <4 x float *> %val, <4 x float *> *%dest
  ret void
}

; Test combinations of vector and non-vector PHIs.
define <4 x float> @f14(<4 x float> %acc, i32 %count) {
; CHECK-LABEL: @f14(
; CHECK: %this_acc.i0 = phi float [ %acc.i0, %entry ], [ %next_acc.i0, %loop ]
; CHECK: %this_acc.i1 = phi float [ %acc.i1, %entry ], [ %next_acc.i1, %loop ]
; CHECK: %this_acc.i2 = phi float [ %acc.i2, %entry ], [ %next_acc.i2, %loop ]
; CHECK: %this_acc.i3 = phi float [ %acc.i3, %entry ], [ %next_acc.i3, %loop ]
; CHECK: %this_count = phi i32 [ %count, %entry ], [ %next_count, %loop ]
; CHECK: %this_acc.upto0 = insertelement <4 x float> poison, float %this_acc.i0, i32 0
; CHECK: %this_acc.upto1 = insertelement <4 x float> %this_acc.upto0, float %this_acc.i1, i32 1
; CHECK: %this_acc.upto2 = insertelement <4 x float> %this_acc.upto1, float %this_acc.i2, i32 2
; CHECK: %this_acc = insertelement <4 x float> %this_acc.upto2, float %this_acc.i3, i32 3
; CHECK: ret <4 x float> %next_acc
entry:
  br label %loop

loop:
  %this_acc = phi <4 x float> [ %acc, %entry ], [ %next_acc, %loop ]
  %this_count = phi i32 [ %count, %entry ], [ %next_count, %loop ]
  %foo = call <4 x float> @ext(<4 x float> %this_acc)
  %next_acc = fadd <4 x float> %this_acc, %foo
  %next_count = sub i32 %this_count, 1
  %cmp = icmp eq i32 %next_count, 0
  br i1 %cmp, label %loop, label %exit

exit:
  ret <4 x float> %next_acc
}

; Test unary operator scalarization.
define void @f15(<4 x float> %init, <4 x float> *%base, i32 %count) {
; CHECK-LABEL: @f15(
; CHECK: %ptr = getelementptr <4 x float>, <4 x float>* %base, i32 %i
; CHECK: %ptr.i0 = bitcast <4 x float>* %ptr to float*
; CHECK: %val.i0 = load float, float* %ptr.i0, align 16
; CHECK: %ptr.i1 = getelementptr float, float* %ptr.i0, i32 1
; CHECK: %val.i1 = load float, float* %ptr.i1, align 4
; CHECK: %ptr.i2 = getelementptr float, float* %ptr.i0, i32 2
; CHECK: %val.i2 = load float, float* %ptr.i2, align 8
; CHECK: %ptr.i3 = getelementptr float, float* %ptr.i0, i32 3
; CHECK: %val.i3 = load float, float* %ptr.i3, align 4
; CHECK: %neg.i0 = fneg float %val.i0
; CHECK: %neg.i1 = fneg float %val.i1
; CHECK: %neg.i2 = fneg float %val.i2
; CHECK: %neg.i3 = fneg float %val.i3
; CHECK: %neg.upto0 = insertelement <4 x float> poison, float %neg.i0, i32 0
; CHECK: %neg.upto1 = insertelement <4 x float> %neg.upto0, float %neg.i1, i32 1
; CHECK: %neg.upto2 = insertelement <4 x float> %neg.upto1, float %neg.i2, i32 2
; CHECK: %neg = insertelement <4 x float> %neg.upto2, float %neg.i3, i32 3
; CHECK: %call = call <4 x float> @ext(<4 x float> %neg)
; CHECK: %call.i0 = extractelement <4 x float> %call, i32 0
; CHECK: %cmp.i0 = fcmp ogt float %call.i0, 1.000000e+00
; CHECK: %call.i1 = extractelement <4 x float> %call, i32 1
; CHECK: %cmp.i1 = fcmp ogt float %call.i1, 2.000000e+00
; CHECK: %call.i2 = extractelement <4 x float> %call, i32 2
; CHECK: %cmp.i2 = fcmp ogt float %call.i2, 3.000000e+00
; CHECK: %call.i3 = extractelement <4 x float> %call, i32 3
; CHECK: %cmp.i3 = fcmp ogt float %call.i3, 4.000000e+00
; CHECK: %sel.i0 = select i1 %cmp.i0, float %call.i0, float 5.000000e+00
; CHECK: %sel.i1 = select i1 %cmp.i1, float %call.i1, float 6.000000e+00
; CHECK: %sel.i2 = select i1 %cmp.i2, float %call.i2, float 7.000000e+00
; CHECK: %sel.i3 = select i1 %cmp.i3, float %call.i3, float 8.000000e+00
; CHECK: store float %sel.i0, float* %ptr.i0, align 16
; CHECK: store float %sel.i1, float* %ptr.i1, align 4
; CHECK: store float %sel.i2, float* %ptr.i2, align 8
; CHECK: store float %sel.i3, float* %ptr.i3, align 4
entry:
  br label %loop

loop:
  %i = phi i32 [ %count, %entry ], [ %nexti, %loop ]
  %acc = phi <4 x float> [ %init, %entry ], [ %sel, %loop ]
  %nexti = sub i32 %i, 1

  %ptr = getelementptr <4 x float>, <4 x float> *%base, i32 %i
  %val = load <4 x float> , <4 x float> *%ptr
  %neg = fneg <4 x float> %val
  %call = call <4 x float> @ext(<4 x float> %neg)
  %cmp = fcmp ogt <4 x float> %call,
  <float 1.0, float 2.0, float 3.0, float 4.0>
  %sel = select <4 x i1> %cmp, <4 x float> %call,
  <4 x float> <float 5.0, float 6.0, float 7.0, float 8.0>
  store <4 x float> %sel, <4 x float> *%ptr

  %test = icmp eq i32 %nexti, 0
  br i1 %test, label %loop, label %exit

exit:
  ret void
}

; Check that IR flags are preserved.
define <2 x i32> @f16(<2 x i32> %i, <2 x i32> %j) {
; CHECK-LABEL: @f16(
; CHECK: %res.i0 = add nuw nsw i32
; CHECK: %res.i1 = add nuw nsw i32
  %res = add nuw nsw <2 x i32> %i, %j
  ret <2 x i32> %res
}
define <2 x i32> @f17(<2 x i32> %i, <2 x i32> %j) {
; CHECK-LABEL: @f17(
; CHECK: %res.i0 = sdiv exact i32
; CHECK: %res.i1 = sdiv exact i32
  %res = sdiv exact <2 x i32> %i, %j
  ret <2 x i32> %res
}
define <2 x float> @f18(<2 x float> %x, <2 x float> %y) {
; CHECK-LABEL: @f18(
; CHECK: %res.i0 = fadd fast float
; CHECK: %res.i1 = fadd fast float
  %res = fadd fast <2 x float> %x, %y
  ret <2 x float> %res
}
define <2 x float> @f19(<2 x float> %x) {
; CHECK-LABEL: @f19(
; CHECK: %res.i0 = fneg fast float
; CHECK: %res.i1 = fneg fast float
  %res = fneg fast <2 x float> %x
  ret <2 x float> %res
}
define <2 x i1> @f20(<2 x float> %x, <2 x float> %y) {
; CHECK-LABEL: @f20(
; CHECK: %res.i0 = fcmp fast ogt float
; CHECK: %res.i1 = fcmp fast ogt float
  %res = fcmp fast ogt <2 x float> %x, %y
  ret <2 x i1> %res
}
declare <2 x float> @llvm.sqrt.v2f32(<2 x float>)
define <2 x float> @f21(<2 x float> %x) {
; CHECK-LABEL: @f21(
; CHECK: %res.i0 = call fast float @llvm.sqrt.f32
; CHECK: %res.i1 = call fast float @llvm.sqrt.f32
  %res = call fast <2 x float> @llvm.sqrt.v2f32(<2 x float> %x)
  ret <2 x float> %res
}
declare <2 x float> @llvm.fma.v2f32(<2 x float>, <2 x float>, <2 x float>)
define <2 x float> @f22(<2 x float> %x, <2 x float> %y, <2 x float> %z) {
; CHECK-LABEL: @f22(
; CHECK: %res.i0 = call fast float @llvm.fma.f32
; CHECK: %res.i1 = call fast float @llvm.fma.f32
  %res = call fast <2 x float> @llvm.fma.v2f32(<2 x float> %x, <2 x float> %y, <2 x float> %z)
  ret <2 x float> %res
}

; See https://reviews.llvm.org/D83101#2133062
define <2 x i32> @f23_crash(<2 x i32> %srcvec, i32 %v1) {
; CHECK-LABEL: @f23_crash(
; CHECK: %v0 = extractelement <2 x i32> %srcvec, i32 0
; CHECK: %t1.upto0 = insertelement <2 x i32> poison, i32 %v0, i32 0
; CHECK: %t1 = insertelement <2 x i32> %t1.upto0, i32 %v1, i32 1
; CHECK: ret <2 x i32> %t1
  %v0 = extractelement <2 x i32> %srcvec, i32 0
  %t0 = insertelement <2 x i32> undef, i32 %v0, i32 0
  %t1 = insertelement <2 x i32> %t0, i32 %v1, i32 1
  ret <2 x i32> %t1
}

!0 = !{ !"root" }
!1 = !{ !"set1", !0 }
!2 = !{ !"set2", !0 }
!3 = !{ !3, !{!"llvm.loop.parallel_accesses", !13} }
!4 = !{ float 4.0 }
!5 = !{ i64 0, i64 8, null }
!13 = distinct !{}
