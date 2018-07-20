; RUN: opt -S -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Check that the vectorizer identifies the %p.09 phi,
; as an induction variable, despite the potential overflow
; due to the truncation from 32bit to 8bit. 
; SCEV will detect the pattern "sext(trunc(%p.09)) + %step"
; and generate the required runtime checks under which
; we can assume no overflow. We check here that we generate
; exactly two runtime checks:
; 1) an overflow check:
;    {0,+,(trunc i32 %step to i8)}<%for.body> Added Flags: <nssw>
; 2) an equality check verifying that the step of the induction 
;    is equal to sext(trunc(step)): 
;    Equal predicate: %step == (sext i8 (trunc i32 %step to i8) to i32)
; 
; See also pr30654.
;
; int a[N];
; void doit1(int n, int step) {
;   int i;
;   char p = 0;
;   for (i = 0; i < n; i++) {
;      a[i] = p;
;      p = p + step;
;   }
; }
; 

; CHECK-LABEL: @doit1
; CHECK: vector.scevcheck
; CHECK: %mul = call { i8, i1 } @llvm.umul.with.overflow.i8(i8 {{.*}}, i8 {{.*}})
; CHECK-NOT: %mul = call { i8, i1 } @llvm.umul.with.overflow.i8(i8 {{.*}}, i8 {{.*}})
; CHECK: %[[TEST:[0-9]+]] = or i1 {{.*}}, %mul.overflow
; CHECK: %[[NTEST:[0-9]+]] = or i1 false, %[[TEST]]
; CHECK: %ident.check = icmp ne i32 {{.*}}, %{{.*}}
; CHECK: %{{.*}} = or i1 %[[NTEST]], %ident.check
; CHECK-NOT: %mul = call { i8, i1 } @llvm.umul.with.overflow.i8(i8 {{.*}}, i8 {{.*}})
; CHECK: vector.body:
; CHECK: <4 x i32>

@a = common local_unnamed_addr global [250 x i32] zeroinitializer, align 16

; Function Attrs: norecurse nounwind uwtable
define void @doit1(i32 %n, i32 %step) local_unnamed_addr {
entry:
  %cmp7 = icmp sgt i32 %n, 0
  br i1 %cmp7, label %for.body.preheader, label %for.end

for.body.preheader:                    
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:                  
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %p.09 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %sext = shl i32 %p.09, 24
  %conv = ashr exact i32 %sext, 24
  %arrayidx = getelementptr inbounds [250 x i32], [250 x i32]* @a, i64 0, i64 %indvars.iv
  store i32 %conv, i32* %arrayidx, align 4
  %add = add nsw i32 %conv, %step
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                        
  br label %for.end

for.end:                                
  ret void
}

; Same as above, but for checking the SCEV "zext(trunc(%p.09)) + %step".
; Here we expect the following two predicates to be added for runtime checking:
; 1) {0,+,(trunc i32 %step to i8)}<%for.body> Added Flags: <nusw>
; 2) Equal predicate: %step == (sext i8 (trunc i32 %step to i8) to i32)
;
; int a[N];
; void doit2(int n, int step) {
;   int i;
;   unsigned char p = 0;
;   for (i = 0; i < n; i++) {
;      a[i] = p;
;      p = p + step;
;   }
; }
; 

; CHECK-LABEL: @doit2
; CHECK: vector.scevcheck
; CHECK: %mul = call { i8, i1 } @llvm.umul.with.overflow.i8(i8 {{.*}}, i8 {{.*}})
; CHECK-NOT: %mul = call { i8, i1 } @llvm.umul.with.overflow.i8(i8 {{.*}}, i8 {{.*}})
; CHECK: %[[TEST:[0-9]+]] = or i1 {{.*}}, %mul.overflow
; CHECK: %[[NTEST:[0-9]+]] = or i1 false, %[[TEST]]
; CHECK: %[[EXT:[0-9]+]] = sext i8 {{.*}} to i32
; CHECK: %ident.check = icmp ne i32 {{.*}}, %[[EXT]]
; CHECK: %{{.*}} = or i1 %[[NTEST]], %ident.check
; CHECK-NOT: %mul = call { i8, i1 } @llvm.umul.with.overflow.i8(i8 {{.*}}, i8 {{.*}})
; CHECK: vector.body:
; CHECK: <4 x i32>

; Function Attrs: norecurse nounwind uwtable
define void @doit2(i32 %n, i32 %step) local_unnamed_addr  {
entry:
  %cmp7 = icmp sgt i32 %n, 0
  br i1 %cmp7, label %for.body.preheader, label %for.end

for.body.preheader:                             
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:                                      
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %p.09 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %conv = and i32 %p.09, 255
  %arrayidx = getelementptr inbounds [250 x i32], [250 x i32]* @a, i64 0, i64 %indvars.iv
  store i32 %conv, i32* %arrayidx, align 4
  %add = add nsw i32 %conv, %step
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                        
  br label %for.end

for.end:                                
  ret void
}

; Here we check that the same phi scev analysis would fail 
; to create the runtime checks because the step is not invariant.
; As a result vectorization will fail.
;
; int a[N];
; void doit3(int n, int step) {
;   int i;
;   char p = 0;
;   for (i = 0; i < n; i++) {
;      a[i] = p;
;      p = p + step;
;      step += 2;
;   }
; }
;

; CHECK-LABEL: @doit3
; CHECK-NOT: vector.scevcheck
; CHECK-NOT: vector.body:
; CHECK-LABEL: for.body:

; Function Attrs: norecurse nounwind uwtable
define void @doit3(i32 %n, i32 %step) local_unnamed_addr {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body.preheader, label %for.end

for.body.preheader:
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %p.012 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %step.addr.010 = phi i32 [ %add3, %for.body ], [ %step, %for.body.preheader ]
  %sext = shl i32 %p.012, 24
  %conv = ashr exact i32 %sext, 24
  %arrayidx = getelementptr inbounds [250 x i32], [250 x i32]* @a, i64 0, i64 %indvars.iv
  store i32 %conv, i32* %arrayidx, align 4
  %add = add nsw i32 %conv, %step.addr.010
  %add3 = add nsw i32 %step.addr.010, 2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}


; Lastly, we also check the case where we can tell at compile time that
; the step of the induction is equal to sext(trunc(step)), in which case
; we don't have to check this equality at runtime (we only need the
; runtime overflow check). Therefore only the following overflow predicate
; will be added for runtime checking:
; {0,+,%cstep}<%for.body> Added Flags: <nssw>
;
; a[N];
; void doit4(int n, char cstep) {
;   int i;
;   char p = 0;
;   int istep = cstep;
;  for (i = 0; i < n; i++) {
;      a[i] = p;
;      p = p + istep;
;   }
; }

; CHECK-LABEL: @doit4
; CHECK: vector.scevcheck
; CHECK: %mul = call { i8, i1 } @llvm.umul.with.overflow.i8(i8 {{.*}}, i8 {{.*}})
; CHECK-NOT: %mul = call { i8, i1 } @llvm.umul.with.overflow.i8(i8 {{.*}}, i8 {{.*}})
; CHECK: %{{.*}} = or i1 {{.*}}, %mul.overflow
; CHECK-NOT: %ident.check = icmp ne i32 {{.*}}, %{{.*}}
; CHECK-NOT: %{{.*}} = or i1 %{{.*}}, %ident.check
; CHECK-NOT: %mul = call { i8, i1 } @llvm.umul.with.overflow.i8(i8 {{.*}}, i8 {{.*}})
; CHECK: vector.body:
; CHECK: <4 x i32>

; Function Attrs: norecurse nounwind uwtable
define void @doit4(i32 %n, i8 signext %cstep) local_unnamed_addr {
entry:
  %conv = sext i8 %cstep to i32
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.end

for.body.preheader:
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %p.011 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %sext = shl i32 %p.011, 24
  %conv2 = ashr exact i32 %sext, 24
  %arrayidx = getelementptr inbounds [250 x i32], [250 x i32]* @a, i64 0, i64 %indvars.iv
  store i32 %conv2, i32* %arrayidx, align 4
  %add = add nsw i32 %conv2, %conv
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}
