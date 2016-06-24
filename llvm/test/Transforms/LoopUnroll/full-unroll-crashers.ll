; Check that we don't crash on corner cases.
; RUN: opt < %s -S -loop-unroll -unroll-max-iteration-count-to-analyze=1000 -unroll-threshold=1 -unroll-percent-dynamic-cost-saved-threshold=20 -o /dev/null
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

@known_constant = internal unnamed_addr constant [10 x i32] [i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1], align 16

define void @foo1() {
entry:
  br label %for.body

for.body:
  %phi = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %idx = zext i32 undef to i64
  %add.ptr = getelementptr inbounds i64, i64* null, i64 %idx
  %inc = add nuw nsw i64 %phi, 1
  %cmp = icmp ult i64 %inc, 999
  br i1 %cmp, label %for.body, label %for.exit

for.exit:
  ret void
}

define void @foo2() {
entry:
  br label %for.body

for.body:
  %phi = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %x = getelementptr i32, <4 x i32*> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %inc = add nuw nsw i64 %phi, 1
  %cmp = icmp ult i64 %inc, 999
  br i1 %cmp, label %for.body, label %for.exit

for.exit:
  ret void
}

define void @cmp_undef() {
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc, %entry
  %iv.0 = phi i64 [ 0, %entry ], [ %iv.1, %for.inc ]
  %arrayidx1 = getelementptr inbounds [10 x i32], [10 x i32]* @known_constant, i64 0, i64 %iv.0
  %x1 = load i32, i32* %arrayidx1, align 4
  %cmp = icmp eq i32 %x1, undef
  br i1 %cmp, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %iv.1 = add nuw nsw i64 %iv.0, 1
  %exitcond = icmp eq i64 %iv.1, 10
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc
  ret void
}

define void @switch() {
entry:
  br label %for.body

for.body:
  %iv.0 = phi i64 [ 0, %entry ], [ %iv.1, %for.inc ]
  %arrayidx1 = getelementptr inbounds [10 x i32], [10 x i32]* @known_constant, i64 0, i64 %iv.0
  %x1 = load i32, i32* %arrayidx1, align 4
  switch i32 %x1, label %l1 [
  ]

l1:
  %x2 = add i32 %x1, 2
  br label %for.inc

for.inc:
  %iv.1 = add nuw nsw i64 %iv.0, 1
  %exitcond = icmp eq i64 %iv.1, 10
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define <4 x i32> @vec_load() {
entry:
  br label %for.body

for.body:
  %phi = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %vec_phi = phi <4 x i32> [ <i32 0, i32 0, i32 0, i32 0>, %entry ], [ %r, %for.body ]
  %arrayidx = getelementptr inbounds [10 x i32], [10 x i32]* @known_constant, i64 0, i64 %phi
  %bc = bitcast i32* %arrayidx to <4 x i32>*
  %x = load <4 x i32>, < 4 x i32>* %bc, align 4
  %r = add <4 x i32> %x, %vec_phi
  %inc = add nuw nsw i64 %phi, 1
  %cmp = icmp ult i64 %inc, 999
  br i1 %cmp, label %for.body, label %for.exit

for.exit:
  ret <4 x i32> %r
}

define void @ptrtoint_cast() optsize {
entry:
  br label %for.body

for.body:
  br i1 true, label %for.inc, label %if.then

if.then:
  %arraydecay = getelementptr inbounds [1 x i32], [1 x i32]* null, i64 0, i64 0
  %x = ptrtoint i32* %arraydecay to i64
  br label %for.inc

for.inc:
  br i1 false, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}

define void @ptrtoint_cast2() {
entry:
  br i1 false, label %for.body.lr.ph, label %exit

for.body.lr.ph:
  br label %for.body

for.body:
  %iv = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %offset = getelementptr inbounds float, float* null, i32 3
  %bc = bitcast float* %offset to i64*
  %inc = add nuw nsw i32 %iv, 1
  br i1 false, label %for.body, label %exit

exit:
  ret void
}

@i = external global i32, align 4

define void @folded_not_to_constantint() {
entry:
  br label %for.body

for.body:
  %iv = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %m = phi i32* [ @i, %entry ], [ %m, %for.inc ]
  br i1 undef, label %if.else, label %if.then

if.then:
  unreachable

if.else:
  %cmp = icmp ult i32* %m, null
  br i1 %cmp, label %cond.false, label %for.inc

cond.false:
  unreachable

for.inc:
  %inc = add nuw nsw i32 %iv, 1
  %cmp2 = icmp ult i32 %inc, 10
  br i1 %cmp2, label %for.body, label %for.end

for.end:
  ret void
}

define void @index_too_large() {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ -73631599, %entry ], [ %iv.next, %for.inc ]
  br i1 undef, label %for.body2, label %for.inc

for.body2:
  %idx = getelementptr inbounds [10 x i32], [10 x i32]* @known_constant, i64 0, i64 %iv
  %x = load i32, i32* %idx, align 1
  br label %for.inc

for.inc:
  %iv.next = add nsw i64 %iv, -1
  br i1 undef, label %for.body, label %for.end

for.end:
  ret void
}

define void @cmp_type_mismatch() {
entry:
  br label %for.header

for.header:
  br label %for.body

for.body:
  %d = phi i32* [ null, %for.header ]
  %cmp = icmp eq i32* %d, null
  br i1 undef, label %for.end, label %for.header

for.end:
  ret void
}

define void @load_type_mismatch() {
entry:
  br label %for.body

for.body:
  %iv.0 = phi i64 [ 0, %entry ], [ %iv.1, %for.body ]
  %arrayidx1 = getelementptr inbounds [10 x i32], [10 x i32]* @known_constant, i64 0, i64 %iv.0
  %bc = bitcast i32* %arrayidx1 to i64*
  %x1 = load i64, i64* %bc, align 4
  %x2 = add i64 10, %x1
  %iv.1 = add nuw nsw i64 %iv.0, 1
  %exitcond = icmp eq i64 %iv.1, 10
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
