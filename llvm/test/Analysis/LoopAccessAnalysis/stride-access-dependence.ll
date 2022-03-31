; RUN: opt -passes='print-access-info' -disable-output  < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

; Following cases are no dependence.

; void nodep_Read_Write(int *A) {
;   int *B = A + 1;
;   for (unsigned i = 0; i < 1024; i+=3)
;     B[i] = A[i] + 1;
; }

; CHECK: function 'nodep_Read_Write':
; CHECK-NEXT:   for.body:
; CHECK-NEXT:     Memory dependences are safe
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:     Run-time memory checks:

define void @nodep_Read_Write(i32* nocapture %A) {
entry:
  %add.ptr = getelementptr inbounds i32, i32* %A, i64 1
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %add.ptr, i64 %indvars.iv
  store i32 %add, i32* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 3
  %cmp = icmp ult i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; int nodep_Write_Read(int *A) {
;   int sum = 0;
;   for (unsigned i = 0; i < 1024; i+=4) {
;     A[i] = i;
;     sum += A[i+3];
;   }
;   
;   return sum;
; }

; CHECK: function 'nodep_Write_Read':
; CHECK-NEXT:   for.body:
; CHECK-NEXT:     Memory dependences are safe
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:     Run-time memory checks:

define i32 @nodep_Write_Read(i32* nocapture %A) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i32 %add3

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %sum.013 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %0 = trunc i64 %indvars.iv to i32
  store i32 %0, i32* %arrayidx, align 4
  %1 = or i64 %indvars.iv, 3
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %1
  %2 = load i32, i32* %arrayidx2, align 4
  %add3 = add nsw i32 %2, %sum.013
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 4
  %cmp = icmp ult i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; void nodep_Write_Write(int *A) {
;   for (unsigned i = 0; i < 1024; i+=2) {
;     A[i] = i;
;     A[i+1] = i+1;
;   }
; }

; CHECK: function 'nodep_Write_Write':
; CHECK-NEXT:   for.body:
; CHECK-NEXT:     Memory dependences are safe
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:     Run-time memory checks:

define void @nodep_Write_Write(i32* nocapture %A) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %0 = trunc i64 %indvars.iv to i32
  store i32 %0, i32* %arrayidx, align 4
  %1 = or i64 %indvars.iv, 1
  %arrayidx3 = getelementptr inbounds i32, i32* %A, i64 %1
  %2 = trunc i64 %1 to i32
  store i32 %2, i32* %arrayidx3, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %cmp = icmp ult i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; Following cases are unsafe depdences and are not vectorizable.

; void unsafe_Read_Write(int *A) {
;   for (unsigned i = 0; i < 1024; i+=3)
;     A[i+3] = A[i] + 1;
; }

; CHECK: function 'unsafe_Read_Write':
; CHECK-NEXT:   for.body:
; CHECK-NEXT:     Report: unsafe dependent memory operations in loop
; CHECK-NEXT:     Backward loop carried data dependence.
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:      Backward:
; CHECK-NEXT:           %0 = load i32, i32* %arrayidx, align 4 -> 
; CHECK-NEXT:           store i32 %add, i32* %arrayidx3, align 4

define void @unsafe_Read_Write(i32* nocapture %A) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %i.010 = phi i32 [ 0, %entry ], [ %add1, %for.body ]
  %idxprom = zext i32 %i.010 to i64
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, 1
  %add1 = add i32 %i.010, 3
  %idxprom2 = zext i32 %add1 to i64
  %arrayidx3 = getelementptr inbounds i32, i32* %A, i64 %idxprom2
  store i32 %add, i32* %arrayidx3, align 4
  %cmp = icmp ult i32 %add1, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; int unsafe_Write_Read(int *A) {
;   int sum = 0;
;   for (unsigned i = 0; i < 1024; i+=4) {
;     A[i] = i;
;     sum += A[i+4];
;   }
;
;   return sum;
; }

; CHECK: function 'unsafe_Write_Read':
; CHECK-NEXT:   for.body:
; CHECK-NEXT:     Report: unsafe dependent memory operations in loop
; CHECK-NEXT:     Backward loop carried data dependence.
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:      Backward:
; CHECK-NEXT:           store i32 %0, i32* %arrayidx, align 4 ->
; CHECK-NEXT:           %1 = load i32, i32* %arrayidx2, align 4

define i32 @unsafe_Write_Read(i32* nocapture %A) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i32 %add3

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %sum.013 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %0 = trunc i64 %indvars.iv to i32
  store i32 %0, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 4
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv.next
  %1 = load i32, i32* %arrayidx2, align 4
  %add3 = add nsw i32 %1, %sum.013
  %cmp = icmp ult i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; void unsafe_Write_Write(int *A) {
;   for (unsigned i = 0; i < 1024; i+=2) {
;     A[i] = i;
;     A[i+2] = i+1;
;   }
; }

; CHECK: function 'unsafe_Write_Write':
; CHECK-NEXT:   for.body:
; CHECK-NEXT:     Report: unsafe dependent memory operations in loop
; CHECK-NEXT:     Backward loop carried data dependence.
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:      Backward:
; CHECK-NEXT:           store i32 %0, i32* %arrayidx, align 4 ->
; CHECK-NEXT:           store i32 %2, i32* %arrayidx3, align 4

define void @unsafe_Write_Write(i32* nocapture %A) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %0 = trunc i64 %indvars.iv to i32
  store i32 %0, i32* %arrayidx, align 4
  %1 = or i64 %indvars.iv, 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %arrayidx3 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv.next
  %2 = trunc i64 %1 to i32
  store i32 %2, i32* %arrayidx3, align 4
  %cmp = icmp ult i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; Following cases check that strided accesses can be vectorized.

; void vectorizable_Read_Write(int *A) {
;   int *B = A + 4;
;   for (unsigned i = 0; i < 1024; i+=2)
;     B[i] = A[i] + 1;
; }

; CHECK: function 'vectorizable_Read_Write':
; CHECK-NEXT:   for.body:
; CHECK-NEXT:     Memory dependences are safe
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:       BackwardVectorizable:
; CHECK-NEXT:           %0 = load i32, i32* %arrayidx, align 4 ->
; CHECK-NEXT:           store i32 %add, i32* %arrayidx2, align 4

define void @vectorizable_Read_Write(i32* nocapture %A) {
entry:
  %add.ptr = getelementptr inbounds i32, i32* %A, i64 4
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %add.ptr, i64 %indvars.iv
  store i32 %add, i32* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %cmp = icmp ult i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; int vectorizable_Write_Read(int *A) {
;   int *B = A + 4;
;   int sum = 0;
;   for (unsigned i = 0; i < 1024; i+=2) {
;     A[i] = i;
;     sum += B[i];
;   }
;   
;   return sum;
; }

; CHECK: function 'vectorizable_Write_Read':
; CHECK-NEXT:   for.body:
; CHECK-NEXT:     Memory dependences are safe
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:       BackwardVectorizable:
; CHECK-NEXT:           store i32 %0, i32* %arrayidx, align 4 ->
; CHECK-NEXT:           %1 = load i32, i32* %arrayidx2, align 4

define i32 @vectorizable_Write_Read(i32* nocapture %A) {
entry:
  %add.ptr = getelementptr inbounds i32, i32* %A, i64 4
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i32 %add

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %sum.013 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %0 = trunc i64 %indvars.iv to i32
  store i32 %0, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %add.ptr, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %1, %sum.013
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %cmp = icmp ult i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; void vectorizable_Write_Write(int *A) {
;   int *B = A + 4;
;   for (unsigned i = 0; i < 1024; i+=2) {
;     A[i] = i;
;     B[i] = i+1;
;   }
; }

; CHECK: function 'vectorizable_Write_Write':
; CHECK-NEXT:   for.body:
; CHECK-NEXT:     Memory dependences are safe
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:       BackwardVectorizable:
; CHECK-NEXT:           store i32 %0, i32* %arrayidx, align 4 -> 
; CHECK-NEXT:           store i32 %2, i32* %arrayidx2, align 4

define void @vectorizable_Write_Write(i32* nocapture %A) {
entry:
  %add.ptr = getelementptr inbounds i32, i32* %A, i64 4
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %0 = trunc i64 %indvars.iv to i32
  store i32 %0, i32* %arrayidx, align 4
  %1 = or i64 %indvars.iv, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %add.ptr, i64 %indvars.iv
  %2 = trunc i64 %1 to i32
  store i32 %2, i32* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %cmp = icmp ult i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; void vectorizable_unscaled_Read_Write(int *A) {
;   int *B = (int *)((char *)A + 14);
;   for (unsigned i = 0; i < 1024; i+=2)
;     B[i] = A[i] + 1;
; }

; FIXME: This case looks like previous case @vectorizable_Read_Write. It sould
; be vectorizable.

; CHECK: function 'vectorizable_unscaled_Read_Write':
; CHECK-NEXT:   for.body:
; CHECK-NEXT:     Report: unsafe dependent memory operations in loop
; CHECK-NEXT:     Backward loop carried data dependence that prevents store-to-load forwarding.
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:       BackwardVectorizableButPreventsForwarding:
; CHECK-NEXT:           %2 = load i32, i32* %arrayidx, align 4 ->
; CHECK-NEXT:           store i32 %add, i32* %arrayidx2, align 4

define void @vectorizable_unscaled_Read_Write(i32* nocapture %A) {
entry:
  %0 = bitcast i32* %A to i8*
  %add.ptr = getelementptr inbounds i8, i8* %0, i64 14
  %1 = bitcast i8* %add.ptr to i32*
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %2, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %1, i64 %indvars.iv
  store i32 %add, i32* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %cmp = icmp ult i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; int vectorizable_unscaled_Write_Read(int *A) {
;   int *B = (int *)((char *)A + 17);
;   int sum = 0;
;   for (unsigned i = 0; i < 1024; i+=2) {
;     A[i] = i;
;     sum += B[i];
;   }
; 
;   return sum;
; }

; CHECK: function 'vectorizable_unscaled_Write_Read':
; CHECK-NEXT:   for.body:
; CHECK-NEXT:     Memory dependences are safe
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:       BackwardVectorizable:
; CHECK-NEXT:           store i32 %2, i32* %arrayidx, align 4 -> 
; CHECK-NEXT:           %3 = load i32, i32* %arrayidx2, align 4

define i32 @vectorizable_unscaled_Write_Read(i32* nocapture %A) {
entry:
  %0 = bitcast i32* %A to i8*
  %add.ptr = getelementptr inbounds i8, i8* %0, i64 17
  %1 = bitcast i8* %add.ptr to i32*
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i32 %add

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %sum.013 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %2 = trunc i64 %indvars.iv to i32
  store i32 %2, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %1, i64 %indvars.iv
  %3 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %3, %sum.013
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %cmp = icmp ult i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; void unsafe_unscaled_Read_Write(int *A) {
;   int *B = (int *)((char *)A + 11);
;   for (unsigned i = 0; i < 1024; i+=2)
;     B[i] = A[i] + 1;
; }

; CHECK: function 'unsafe_unscaled_Read_Write':
; CHECK-NEXT:   for.body:
; CHECK-NEXT:     Report: unsafe dependent memory operations in loop
; CHECK-NEXT:     Backward loop carried data dependence.
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:       Backward:
; CHECK-NEXT:           %2 = load i32, i32* %arrayidx, align 4 -> 
; CHECK-NEXT:           store i32 %add, i32* %arrayidx2, align 4

define void @unsafe_unscaled_Read_Write(i32* nocapture %A) {
entry:
  %0 = bitcast i32* %A to i8*
  %add.ptr = getelementptr inbounds i8, i8* %0, i64 11
  %1 = bitcast i8* %add.ptr to i32*
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %2, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %1, i64 %indvars.iv
  store i32 %add, i32* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %cmp = icmp ult i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; CHECK: function 'unsafe_unscaled_Read_Write2':
; CHECK-NEXT:   for.body:
; CHECK-NEXT:     Report: unsafe dependent memory operations in loop
; CHECK-NEXT:     Backward loop carried data dependence.
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:       Backward:
; CHECK-NEXT:           %2 = load i32, i32* %arrayidx, align 4 -> 
; CHECK-NEXT:           store i32 %add, i32* %arrayidx2, align 4

; void unsafe_unscaled_Read_Write2(int *A) {
;   int *B = (int *)((char *)A + 1);
;   for (unsigned i = 0; i < 1024; i+=2)
;     B[i] = A[i] + 1;
; }

define void @unsafe_unscaled_Read_Write2(i32* nocapture %A) {
entry:
  %0 = bitcast i32* %A to i8*
  %add.ptr = getelementptr inbounds i8, i8* %0, i64 1
  %1 = bitcast i8* %add.ptr to i32*
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %2, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %1, i64 %indvars.iv
  store i32 %add, i32* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %cmp = icmp ult i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; Following case checks that interleaved stores have dependences with another
; store and can not pass dependence check.

; void interleaved_stores(int *A) {
;   int *B = (int *) ((char *)A + 1);
;   for(int i = 0; i < 1024; i+=2) {
;     B[i]   = i;                // (1)
;     A[i+1] = i + 1;            // (2)
;     B[i+1] = i + 1;            // (3)
;   }
; }
;
; The access (2) has overlaps with (1) and (3).

; CHECK: function 'interleaved_stores':
; CHECK-NEXT:   for.body:
; CHECK-NEXT:     Report: unsafe dependent memory operations in loop
; CHECK-NEXT:     Backward loop carried data dependence.
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:       Backward:
; CHECK-NEXT:           store i32 %4, i32* %arrayidx5, align 4 -> 
; CHECK-NEXT:           store i32 %4, i32* %arrayidx9, align 4
; CHECK:       Backward:
; CHECK-NEXT:           store i32 %2, i32* %arrayidx2, align 4 -> 
; CHECK-NEXT:           store i32 %4, i32* %arrayidx5, align 4

define void @interleaved_stores(i32* nocapture %A) {
entry:
  %0 = bitcast i32* %A to i8*
  %incdec.ptr = getelementptr inbounds i8, i8* %0, i64 1
  %1 = bitcast i8* %incdec.ptr to i32*
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %2 = trunc i64 %indvars.iv to i32
  %arrayidx2 = getelementptr inbounds i32, i32* %1, i64 %indvars.iv
  store i32 %2, i32* %arrayidx2, align 4
  %3 = or i64 %indvars.iv, 1
  %arrayidx5 = getelementptr inbounds i32, i32* %A, i64 %3
  %4 = trunc i64 %3 to i32
  store i32 %4, i32* %arrayidx5, align 4
  %arrayidx9 = getelementptr inbounds i32, i32* %1, i64 %3
  store i32 %4, i32* %arrayidx9, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %cmp = icmp slt i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}
