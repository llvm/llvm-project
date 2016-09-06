; RUN: opt -mtriple=aarch64--linux-gnueabi -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 < %s -S | FileCheck %s

; The following tests contain loops for which SCEV cannot determine the backedge
; taken count. This is because the backedge taken condition is produced by an
; icmp with one of the sides being a loop varying non-AddRec expression.
; However, there is a possibility to normalize this to an AddRec expression
; using SCEV predicates. This allows us to compute a 'guarded' backedge count.
; The Loop Vectorizer is able to version to loop in order to use this guarded
; backedge count and vectorize more loops.


; CHECK-LABEL: test_sge
; CHECK-LABEL: vector.scevcheck
; CHECK-LABEL: vector.body
define void @test_sge(i32* noalias %A,
                      i32* noalias %B,
                      i32* noalias %C, i32 %N) {
entry:
  %cmp13 = icmp eq i32 %N, 0
  br i1 %cmp13, label %for.end, label %for.body.preheader

for.body.preheader:
  br label %for.body

for.body:
  %indvars.iv = phi i16 [ %indvars.next, %for.body ], [ 0, %for.body.preheader ]
  %indvars.next = add i16 %indvars.iv, 1
  %indvars.ext = zext i16 %indvars.iv to i32

  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %indvars.ext
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %C, i32 %indvars.ext
  %1 = load i32, i32* %arrayidx3, align 4

  %mul4 = mul i32 %1, %0

  %arrayidx7 = getelementptr inbounds i32, i32* %A, i32 %indvars.ext
  store i32 %mul4, i32* %arrayidx7, align 4

  %exitcond = icmp sge i32 %indvars.ext, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

; CHECK-LABEL: test_uge
; CHECK-LABEL: vector.scevcheck
; CHECK-LABEL: vector.body
define void @test_uge(i32* noalias %A,
                      i32* noalias %B,
                      i32* noalias %C, i32 %N, i32 %Offset) {
entry:
  %cmp13 = icmp eq i32 %N, 0
  br i1 %cmp13, label %for.end, label %for.body.preheader

for.body.preheader:
  br label %for.body

for.body:
  %indvars.iv = phi i16 [ %indvars.next, %for.body ], [ 0, %for.body.preheader ]
  %indvars.next = add i16 %indvars.iv, 1

  %indvars.ext = sext i16 %indvars.iv to i32
  %indvars.access = add i32 %Offset, %indvars.ext

  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %indvars.access
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %C, i32 %indvars.access
  %1 = load i32, i32* %arrayidx3, align 4

  %mul4 = add i32 %1, %0

  %arrayidx7 = getelementptr inbounds i32, i32* %A, i32 %indvars.access
  store i32 %mul4, i32* %arrayidx7, align 4

  %exitcond = icmp uge i32 %indvars.ext, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

; CHECK-LABEL: test_ule
; CHECK-LABEL: vector.scevcheck
; CHECK-LABEL: vector.body
define void @test_ule(i32* noalias %A,
                      i32* noalias %B,
                      i32* noalias %C, i32 %N,
                      i16 %M) {
entry:
  %cmp13 = icmp eq i32 %N, 0
  br i1 %cmp13, label %for.end, label %for.body.preheader

for.body.preheader:
  br label %for.body

for.body:
  %indvars.iv = phi i16 [ %indvars.next, %for.body ], [ %M, %for.body.preheader ]
  %indvars.next = sub i16 %indvars.iv, 1
  %indvars.ext = zext i16 %indvars.iv to i32

  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %indvars.ext
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %C, i32 %indvars.ext
  %1 = load i32, i32* %arrayidx3, align 4

  %mul4 = mul i32 %1, %0

  %arrayidx7 = getelementptr inbounds i32, i32* %A, i32 %indvars.ext
  store i32 %mul4, i32* %arrayidx7, align 4

  %exitcond = icmp ule i32 %indvars.ext, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

; CHECK-LABEL: test_sle
; CHECK-LABEL: vector.scevcheck
; CHECK-LABEL: vector.body
define void @test_sle(i32* noalias %A,
                   i32* noalias %B,
                   i32* noalias %C, i32 %N,
                   i16 %M) {
entry:
  %cmp13 = icmp eq i32 %N, 0
  br i1 %cmp13, label %for.end, label %for.body.preheader

for.body.preheader:
  br label %for.body

for.body:
  %indvars.iv = phi i16 [ %indvars.next, %for.body ], [ %M, %for.body.preheader ]
  %indvars.next = sub i16 %indvars.iv, 1
  %indvars.ext = sext i16 %indvars.iv to i32

  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %indvars.ext
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %C, i32 %indvars.ext
  %1 = load i32, i32* %arrayidx3, align 4

  %mul4 = mul i32 %1, %0

  %arrayidx7 = getelementptr inbounds i32, i32* %A, i32 %indvars.ext
  store i32 %mul4, i32* %arrayidx7, align 4

  %exitcond = icmp sle i32 %indvars.ext, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}
