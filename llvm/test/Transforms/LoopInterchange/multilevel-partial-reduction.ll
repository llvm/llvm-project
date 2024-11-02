; RUN: opt < %s -loop-interchange -cache-line-size=4 -pass-remarks-missed='loop-interchange' -pass-remarks-output=%t -S \
; RUN:     -verify-dom-info -verify-loop-info -verify-loop-lcssa
; RUN: FileCheck --input-file=%t --check-prefix=REMARKS %s

@b = external global [512 x [4 x i32]]
@c = global [2 x [4 x i32]] zeroinitializer, align 1

; Check that the outermost and the middle loops are not interchanged since
; the innermost loop has a reduction operation which is however not in a form
; that loop interchange can handle. Interchanging the outermost and the
; middle loops would intervene with the reduction and cause miscompile.

; REMARKS: --- !Missed
; REMARKS-NEXT: Pass:            loop-interchange
; REMARKS-NEXT: Name:            UnsupportedPHIInner
; REMARKS-NEXT: Function:        test7
; REMARKS: --- !Missed
; REMARKS-NEXT: Pass:            loop-interchange
; REMARKS-NEXT: Name:            UnsupportedPHIInner
; REMARKS-NEXT: Function:        test7

define i32 @test7() {
entry:
  br label %for.cond1.preheader.i

for.cond1.preheader.i:                            ; preds = %for.inc19.i, %entry
  %i.011.i = phi i16 [ 0, %entry ], [ %inc20.i, %for.inc19.i ]
  br label %for.cond4.preheader.i

for.cond4.preheader.i:                            ; preds = %middle.block, %for.cond1.preheader.i
  %j.010.i = phi i16 [ 0, %for.cond1.preheader.i ], [ %inc17.i, %middle.block ]
  %arrayidx14.i = getelementptr inbounds [2 x [4 x i32]], ptr @c, i16 0, i16 %i.011.i, i16 %j.010.i
  %arrayidx14.promoted.i = load i32, ptr %arrayidx14.i, align 1
  %0 = insertelement <4 x i32> <i32 poison, i32 0, i32 0, i32 0>, i32 %arrayidx14.promoted.i, i64 0
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %for.cond4.preheader.i
  %index = phi i16 [ 0, %for.cond4.preheader.i ], [ %index.next, %vector.body ]
  %vec.phi = phi <4 x i32> [ %0, %for.cond4.preheader.i ], [ %16, %vector.body ]
  %1 = or i16 %index, 1
  %2 = or i16 %index, 2
  %3 = or i16 %index, 3
  %4 = getelementptr inbounds [512 x [4 x i32]], ptr @b, i16 0, i16 %index, i16 %j.010.i
  %5 = getelementptr inbounds [512 x [4 x i32]], ptr @b, i16 0, i16 %1, i16 %j.010.i
  %6 = getelementptr inbounds [512 x [4 x i32]], ptr @b, i16 0, i16 %2, i16 %j.010.i
  %7 = getelementptr inbounds [512 x [4 x i32]], ptr @b, i16 0, i16 %3, i16 %j.010.i
  %8 = load i32, ptr %4, align 1
  %9 = load i32, ptr %5, align 1
  %10 = load i32, ptr %6, align 1
  %11 = load i32, ptr %7, align 1
  %12 = insertelement <4 x i32> poison, i32 %8, i64 0
  %13 = insertelement <4 x i32> %12, i32 %9, i64 1
  %14 = insertelement <4 x i32> %13, i32 %10, i64 2
  %15 = insertelement <4 x i32> %14, i32 %11, i64 3
  %16 = add <4 x i32> %15, %vec.phi
  %index.next = add nuw i16 %index, 4
  %17 = icmp eq i16 %index.next, 512
  br i1 %17, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %18 = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %16)
  store i32 %18, ptr %arrayidx14.i, align 1
  %inc17.i = add nuw nsw i16 %j.010.i, 1
  %exitcond12.not.i = icmp eq i16 %inc17.i, 4
  br i1 %exitcond12.not.i, label %for.inc19.i, label %for.cond4.preheader.i

for.inc19.i:                                      ; preds = %middle.block
  %inc20.i = add nuw nsw i16 %i.011.i, 1
  %exitcond13.not.i = icmp eq i16 %inc20.i, 2
  br i1 %exitcond13.not.i, label %test.exit, label %for.cond1.preheader.i

test.exit:                                        ; preds = %for.inc19.i
  %19 = load i32, ptr @c, align 1
  ret i32 %19
}

declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>)
