; RUN: opt < %s -mattr=+sve -vector-library=ArmPL -passes=inject-tli-mappings,loop-vectorize -debug-only=loop-accesses -disable-output 2>&1 | FileCheck %s

; REQUIRES: asserts

target triple = "aarch64-unknown-linux-gnu"

; TODO: add mappings for frexp/frexpf

define void @frexp_f64(ptr %in, ptr %out1, ptr %out2, i32 %N) {
entry:
  %cmp4 = icmp sgt i32 %N, 0
  br i1 %cmp4, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %wide.trip.count = zext nneg i32 %N to i64
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %in, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %add.ptr = getelementptr inbounds i32, ptr %out2, i64 %indvars.iv
  %call = tail call double @frexp(double noundef %0, ptr noundef %add.ptr)
  store double %call, ptr %out1, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @frexp(double, ptr) #1

define void @frexp_f32(ptr readonly %in, ptr %out1, ptr %out2, i32 %N) {
entry:
  %cmp4 = icmp sgt i32 %N, 0
  br i1 %cmp4, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %wide.trip.count = zext nneg i32 %N to i64
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %in, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %add.ptr = getelementptr inbounds i32, ptr %out2, i64 %indvars.iv
  %call = tail call float @frexpf(float noundef %0, ptr noundef %add.ptr)
  store float %call, ptr %out1, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare float @frexpf(float , ptr) #1

define void @modf_f64(ptr %in, ptr %out1, ptr %out2, i32 %N) {
; CHECK: LAA: allow math function with write-only attribute:  %call = tail call double @modf
entry:
  %cmp7 = icmp sgt i32 %N, 0
  br i1 %cmp7, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %wide.trip.count = zext nneg i32 %N to i64
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %in, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %add.ptr = getelementptr inbounds double, ptr %out2, i64 %indvars.iv
  %call = tail call double @modf(double noundef %0, ptr noundef %add.ptr)
  %arrayidx2 = getelementptr inbounds double, ptr %out1, i64 %indvars.iv
  store double %call, ptr %arrayidx2, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @modf(double , ptr ) #1

define void @modf_f32(ptr %in, ptr %out1, ptr %out2, i32 %N) {
; CHECK: LAA: allow math function with write-only attribute:  %call = tail call float @modff
entry:
  %cmp7 = icmp sgt i32 %N, 0
  br i1 %cmp7, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %wide.trip.count = zext nneg i32 %N to i64
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %in, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %add.ptr = getelementptr inbounds float, ptr %out2, i64 %indvars.iv
  %call = tail call float @modff(float noundef %0, ptr noundef %add.ptr)
  %arrayidx2 = getelementptr inbounds float, ptr %out1, i64 %indvars.iv
  store float %call, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare float @modff(float noundef, ptr nocapture noundef) #1

attributes #1 = { memory(argmem: write) }