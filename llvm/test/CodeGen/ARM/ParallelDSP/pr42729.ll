; RUN: opt -mtriple=thumbv7-unknown-linux-android -arm-parallel-dsp -S %s -o - | FileCheck %s

; CHECK-LABEL: undef_no_return
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %incdec.ptr21 to i32*
; CHECK: [[LOAD_A:%[^ ]+]] = load i32, i32* [[CAST_A]], align 2
; CHECK: %uglygep15 = getelementptr i8, i8* undef, i32 undef
; CHECK: [[GEP8:%[^ ]+]] = getelementptr i8, i8* undef, i32 undef
; CHECK: [[CAST_GEP8:%[^ ]+]] = bitcast i8* [[GEP8]] to i16*
; CHECK: [[GEP16:%[^ ]+]] = getelementptr i16, i16* [[CAST_GEP8]], i32 6
; CHECK: [[CAST_GEP16:%[^ ]+]] = bitcast i16* [[GEP16]] to i32*
; CHECK: [[LOAD_UNDEF:%[^ ]+]] = load i32, i32* [[CAST_GEP16]], align 2
; CHECK: call i32 @llvm.arm.smladx(i32 [[LOAD_A]], i32 [[LOAD_UNDEF]], i32 undef)
define void @undef_no_return(i16* %a) {
entry:
  %incdec.ptr21 = getelementptr inbounds i16, i16* %a, i32 3
  %incdec.ptr29 = getelementptr inbounds i16, i16* %a, i32 4
  br label %for.body

for.body:
  %0 = load i16, i16* %incdec.ptr21, align 2
  %conv25 = sext i16 %0 to i32
  %uglygep15 = getelementptr i8, i8* undef, i32 undef
  %uglygep1516 = bitcast i8* %uglygep15 to i16*
  %scevgep17 = getelementptr i16, i16* %uglygep1516, i32 7
  %1 = load i16, i16* %scevgep17, align 2
  %conv31 = sext i16 %1 to i32
  %2 = load i16, i16* %incdec.ptr29, align 2
  %conv33 = sext i16 %2 to i32
  %uglygep12 = getelementptr i8, i8* undef, i32 undef
  %uglygep1213 = bitcast i8* %uglygep12 to i16*
  %scevgep14 = getelementptr i16, i16* %uglygep1213, i32 6
  %3 = load i16, i16* %scevgep14, align 2
  %conv39 = sext i16 %3 to i32
  %mul.i287.neg.neg = mul nsw i32 %conv31, %conv25
  %mul.i283.neg.neg = mul nsw i32 %conv39, %conv33
  %reass.add408 = add i32 undef, %mul.i287.neg.neg
  %reass.add409 = add i32 %reass.add408, %mul.i283.neg.neg
  br label %for.body
}

; CHECK-LABEL: return
; CHECK: phi i32 [ %N, %entry ]
; CHECK: [[ACC:%[^ ]+]] = phi i32 [ 0, %entry ], [ [[ACC_NEXT:%[^ ]+]], %for.body ]
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %incdec.ptr21 to i32*
; CHECK: [[LOAD_A:%[^ ]+]] = load i32, i32* [[CAST_A]], align 2
; CHECK: [[GEP8:%[^ ]+]] = getelementptr i8, i8* %b, i32 0
; CHECK: [[CAST_GEP8:%[^ ]+]] = bitcast i8* [[GEP8]] to i16*
; CHECK: [[GEP16:%[^ ]+]] = getelementptr i16, i16* [[CAST_GEP8]], i32 %iv
; CHECK: [[CAST_GEP16:%[^ ]+]] = bitcast i16* [[GEP16]] to i32*
; CHECK: [[LOAD_B:%[^ ]+]] = load i32, i32* [[CAST_GEP16]], align 2
; CHECK: [[ACC_NEXT]] = call i32 @llvm.arm.smladx(i32 [[LOAD_A]], i32 [[LOAD_B]], i32 [[ACC]])
define i32 @return(i16* %a, i8* %b, i32 %N) {
entry:
  %incdec.ptr21 = getelementptr inbounds i16, i16* %a, i32 3
  %incdec.ptr29 = getelementptr inbounds i16, i16* %a, i32 4
  br label %for.body

for.body:
  %iv = phi i32 [ %N, %entry ], [ %iv.next, %for.body ]
  %acc = phi i32 [ 0, %entry ], [ %reass.add409, %for.body ]
  %0 = load i16, i16* %incdec.ptr21, align 2
  %conv25 = sext i16 %0 to i32
  %uglygep15 = getelementptr i8, i8* %b, i32 0
  %uglygep1516 = bitcast i8* %uglygep15 to i16*
  %b.idx = add nuw nsw i32 %iv, 1
  %scevgep17 = getelementptr i16, i16* %uglygep1516, i32 %b.idx
  %scevgep14 = getelementptr i16, i16* %uglygep1516, i32 %iv
  %1 = load i16, i16* %scevgep17, align 2
  %conv31 = sext i16 %1 to i32
  %2 = load i16, i16* %incdec.ptr29, align 2
  %conv33 = sext i16 %2 to i32
  %3 = load i16, i16* %scevgep14, align 2
  %conv39 = sext i16 %3 to i32
  %mul.i287.neg.neg = mul nsw i32 %conv31, %conv25
  %mul.i283.neg.neg = mul nsw i32 %conv39, %conv33
  %reass.add408 = add i32 %acc, %mul.i287.neg.neg
  %reass.add409 = add i32 %reass.add408, %mul.i283.neg.neg
  %iv.next = add nuw nsw i32 %iv, -1
  %cmp = icmp ne i32 %iv.next, 0
  br i1 %cmp, label %for.body, label %exit

exit:
  ret i32 %reass.add409
}
