; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s

declare { i16, i1 } @llvm.sadd.with.overflow.i16(i16, i16) nounwind readnone
declare { i16, i1 } @llvm.uadd.with.overflow.i16(i16, i16) nounwind readnone
declare { i16, i1 } @llvm.ssub.with.overflow.i16(i16, i16) nounwind readnone
declare { i16, i1 } @llvm.usub.with.overflow.i16(i16, i16) nounwind readnone
declare { i16, i1 } @llvm.smul.with.overflow.i16(i16, i16) nounwind readnone
declare { i16, i1 } @llvm.umul.with.overflow.i16(i16, i16) nounwind readnone

; CHECK-LABEL: Classifying expressions for: @uadd_exhaustive
; CHECK: Loop %for.body: backedge-taken count is 35
define void @uadd_exhaustive() {
entry:
  br i1 undef, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i16 [ %math, %for.body ], [ 65500, %for.body.preheader ]
  %0 = call { i16, i1 } @llvm.uadd.with.overflow.i16(i16 %indvars.iv, i16 1)
  %math = extractvalue { i16, i1 } %0, 0
  %ov = extractvalue { i16, i1 } %0, 1
  br i1 %ov, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; CHECK-LABEL: Classifying expressions for: @sadd_exhaustive
; CHECK: Loop %for.body: backedge-taken count is 67
define void @sadd_exhaustive() {
entry:
  br i1 undef, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i16 [ %math, %for.body ], [ 32700, %for.body.preheader ]
  %0 = call { i16, i1 } @llvm.sadd.with.overflow.i16(i16 %indvars.iv, i16 1)
  %math = extractvalue { i16, i1 } %0, 0
  %ov = extractvalue { i16, i1 } %0, 1
  br i1 %ov, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; CHECK-LABEL: Classifying expressions for: @usub_exhaustive
; CHECK: Loop %for.body: backedge-taken count is 50
define void @usub_exhaustive() {
entry:
  br i1 undef, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i16 [ %math, %for.body ], [ 50, %for.body.preheader ]
  %0 = call { i16, i1 } @llvm.usub.with.overflow.i16(i16 %indvars.iv, i16 1)
  %math = extractvalue { i16, i1 } %0, 0
  %ov = extractvalue { i16, i1 } %0, 1
  br i1 %ov, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; CHECK-LABEL: Classifying expressions for: @ssub_exhaustive
; CHECK: Loop %for.body: backedge-taken count is 68
define void @ssub_exhaustive() {
entry:
  br i1 undef, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i16 [ %math, %for.body ], [ -32700, %for.body.preheader ]
  %0 = call { i16, i1 } @llvm.ssub.with.overflow.i16(i16 %indvars.iv, i16 1)
  %math = extractvalue { i16, i1 } %0, 0
  %ov = extractvalue { i16, i1 } %0, 1
  br i1 %ov, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; CHECK-LABEL: Classifying expressions for: @smul_exhaustive
; CHECK: Loop %for.body: backedge-taken count is 14
define void @smul_exhaustive() {
entry:
  br i1 undef, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i16 [ %math, %for.body ], [ 1, %for.body.preheader ]
  %0 = call { i16, i1 } @llvm.smul.with.overflow.i16(i16 %indvars.iv, i16 2)
  %math = extractvalue { i16, i1 } %0, 0
  %ov = extractvalue { i16, i1 } %0, 1
  br i1 %ov, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; CHECK-LABEL: Classifying expressions for: @umul_exhaustive
; CHECK: Loop %for.body: backedge-taken count is 15
define void @umul_exhaustive() {
entry:
  br i1 undef, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i16 [ %math, %for.body ], [ 1, %for.body.preheader ]
  %0 = call { i16, i1 } @llvm.umul.with.overflow.i16(i16 %indvars.iv, i16 2)
  %math = extractvalue { i16, i1 } %0, 0
  %ov = extractvalue { i16, i1 } %0, 1
  br i1 %ov, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}
