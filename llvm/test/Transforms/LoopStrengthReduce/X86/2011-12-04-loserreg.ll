; RUN: opt < %s -loop-reduce -S | FileCheck %s
;
; Test LSR's ability to prune formulae that refer to nonexistent
; AddRecs in other loops.
;
; Unable to reduce this case further because it requires LSR to exceed
; ComplexityLimit.
;
; We really just want to ensure that LSR can process this loop without
; finding an unsatisfactory solution and bailing out. I've added
; dummyout, an obvious candidate for postinc replacement so we can
; verify that LSR removes it.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

; CHECK-LABEL: @test(
; CHECK: for.body:
; CHECK: %lsr.iv
; CHECK-NOT: %dummyout
; CHECK: ret
define i64 @test(i64 %count, float* nocapture %srcrow, i32* nocapture %destrow) nounwind uwtable ssp {
entry:
  %cmp34 = icmp eq i64 %count, 0
  br i1 %cmp34, label %for.end29, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %dummyiv = phi i64 [ %dummycnt, %for.body ], [ 0, %entry ]
  %indvars.iv39 = phi i64 [ %indvars.iv.next40, %for.body ], [ 0, %entry ]
  %dp.036 = phi i32* [ %add.ptr, %for.body ], [ %destrow, %entry ]
  %p.035 = phi float* [ %incdec.ptr4, %for.body ], [ %srcrow, %entry ]
  %incdec.ptr = getelementptr inbounds float, float* %p.035, i64 1
  %0 = load float, float* %incdec.ptr, align 4
  %incdec.ptr2 = getelementptr inbounds float, float* %p.035, i64 2
  %1 = load float, float* %incdec.ptr2, align 4
  %incdec.ptr3 = getelementptr inbounds float, float* %p.035, i64 3
  %2 = load float, float* %incdec.ptr3, align 4
  %incdec.ptr4 = getelementptr inbounds float, float* %p.035, i64 4
  %3 = load float, float* %incdec.ptr4, align 4
  %4 = load i32, i32* %dp.036, align 4
  %conv5 = fptoui float %0 to i32
  %or = or i32 %4, %conv5
  %arrayidx6 = getelementptr inbounds i32, i32* %dp.036, i64 1
  %5 = load i32, i32* %arrayidx6, align 4
  %conv7 = fptoui float %1 to i32
  %or8 = or i32 %5, %conv7
  %arrayidx9 = getelementptr inbounds i32, i32* %dp.036, i64 2
  %6 = load i32, i32* %arrayidx9, align 4
  %conv10 = fptoui float %2 to i32
  %or11 = or i32 %6, %conv10
  %arrayidx12 = getelementptr inbounds i32, i32* %dp.036, i64 3
  %7 = load i32, i32* %arrayidx12, align 4
  %conv13 = fptoui float %3 to i32
  %or14 = or i32 %7, %conv13
  store i32 %or, i32* %dp.036, align 4
  store i32 %or8, i32* %arrayidx6, align 4
  store i32 %or11, i32* %arrayidx9, align 4
  store i32 %or14, i32* %arrayidx12, align 4
  %add.ptr = getelementptr inbounds i32, i32* %dp.036, i64 4
  %indvars.iv.next40 = add i64 %indvars.iv39, 4
  %dummycnt = add i64 %dummyiv, 1
  %cmp = icmp ult i64 %indvars.iv.next40, %count
  br i1 %cmp, label %for.body, label %for.cond19.preheader

for.cond19.preheader:                             ; preds = %for.body
  %dummyout = add i64 %dummyiv, 1
  %rem = and i64 %count, 3
  %cmp2130 = icmp eq i64 %rem, 0
  br i1 %cmp2130, label %for.end29, label %for.body23.lr.ph

for.body23.lr.ph:                                 ; preds = %for.cond19.preheader
  %8 = and i64 %count, 3
  br label %for.body23

for.body23:                                       ; preds = %for.body23, %for.body23.lr.ph
  %indvars.iv = phi i64 [ 0, %for.body23.lr.ph ], [ %indvars.iv.next, %for.body23 ]
  %dp.132 = phi i32* [ %add.ptr, %for.body23.lr.ph ], [ %incdec.ptr28, %for.body23 ]
  %p.131 = phi float* [ %incdec.ptr4, %for.body23.lr.ph ], [ %incdec.ptr24, %for.body23 ]
  %incdec.ptr24 = getelementptr inbounds float, float* %p.131, i64 1
  %9 = load float, float* %incdec.ptr24, align 4
  %10 = load i32, i32* %dp.132, align 4
  %conv25 = fptoui float %9 to i32
  %or26 = or i32 %10, %conv25
  store i32 %or26, i32* %dp.132, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %incdec.ptr28 = getelementptr inbounds i32, i32* %dp.132, i64 1
  %exitcond = icmp eq i64 %indvars.iv.next, %8
  br i1 %exitcond, label %for.end29, label %for.body23

for.end29:                                        ; preds = %entry, %for.body23, %for.cond19.preheader
  %result = phi i64 [ 0, %entry ], [ %dummyout, %for.body23 ], [ %dummyout, %for.cond19.preheader ]
  ret i64 %result
}
