; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 \
; RUN:     -pass-remarks-output=%t -disable-output -loop-interchange-profitabilities=vectorize
; RUN: FileCheck -input-file %t %s

@A = dso_local global [256 x [256 x float]] zeroinitializer
@B = dso_local global [256 x [256 x float]] zeroinitializer
@C = dso_local global [256 x [256 x float]] zeroinitializer
@D = global [256 x [256 x [256 x float]]] zeroinitializer
@E = global [256 x [256 x [256 x float]]] zeroinitializer

; Check that the below loops are exchanged for vectorization.
;
; for (int i = 0; i < 256; i++) {
;   for (int j = 1; j < 256; j++) {
;     A[i][j] = A[i][j-1] + B[i][j];
;     C[i][j] += 1;
;   }
; }
;

; CHECK:      --- !Passed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        interchange_necessary_for_vectorization
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Loop interchanged with enclosing loop.
define void @interchange_necessary_for_vectorization() {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 1, %entry ], [ %i.next, %for.i.inc ]
  br label %for.j.body

for.j.body:
  %j = phi i64 [ 1, %for.i.header ], [ %j.next, %for.j.body ]
  %j.dec = add nsw i64 %j, -1
  %a.load.index = getelementptr nuw inbounds [256 x [256 x float]], ptr @A, i64 0, i64 %i, i64 %j.dec
  %b.index = getelementptr nuw inbounds [256 x [256 x float]], ptr @B, i64 0, i64 %i, i64 %j
  %c.index = getelementptr nuw inbounds [256 x [256 x float]], ptr @C, i64 0, i64 %i, i64 %j
  %a = load float, ptr %a.load.index, align 4
  %b = load float, ptr %b.index, align 4
  %c = load float, ptr %c.index, align 4
  %add.0 = fadd float %a, %b
  %a.store.index = getelementptr nuw inbounds [256 x [256 x float]], ptr @A, i64 0, i64 %i, i64 %j
  store float %add.0, ptr %a.store.index, align 4
  %add.1 = fadd float %c, 1.0
  store float %add.1, ptr %c.index, align 4
  %j.next = add nuw nsw i64 %j, 1
  %cmp.j = icmp eq i64 %j.next, 256
  br i1 %cmp.j, label %for.i.inc, label %for.j.body

for.i.inc:
  %i.next = add nuw nsw i64 %i, 1
  %cmp.i = icmp eq i64 %i.next, 256
  br i1 %cmp.i, label %exit, label %for.i.header

exit:
  ret void
}

; Check that the following innermost loop can be vectorized so that
; interchanging is unnecessary.
;
; for (int i = 0; i < 256; i++)
;   for (int j = 1; j < 256; j++)
;     A[i][j-1] = A[i][j] + B[i][j];
;

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            InterchangeNotProfitable
; CHECK-NEXT: Function:        interchange_unnecesasry_for_vectorization
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Insufficient information to calculate the cost of loop for interchange.
define void @interchange_unnecesasry_for_vectorization() {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 1, %entry ], [ %i.next, %for.i.inc ]
  br label %for.j.body

for.j.body:
  %j = phi i64 [ 1, %for.i.header ], [ %j.next, %for.j.body ]
  %j.dec = add nsw i64 %j, -1
  %a.load.index = getelementptr nuw inbounds [256 x [256 x float]], ptr @A, i64 0, i64 %i, i64 %j
  %b.index = getelementptr nuw inbounds [256 x [256 x float]], ptr @B, i64 0, i64 %i, i64 %j
  %a = load float, ptr %a.load.index, align 4
  %b = load float, ptr %b.index, align 4
  %add = fadd float %a, %b
  %a.store.index = getelementptr nuw inbounds [256 x [256 x float]], ptr @A, i64 0, i64 %i, i64 %j.dec
  store float %add, ptr %a.store.index, align 4
  %j.next = add nuw nsw i64 %j, 1
  %cmp.j = icmp eq i64 %j.next, 256
  br i1 %cmp.j, label %for.i.inc, label %for.j.body

for.i.inc:
  %i.next = add nuw nsw i64 %i, 1
  %cmp.i = icmp eq i64 %i.next, 256
  br i1 %cmp.i, label %exit, label %for.i.header

exit:
  ret void
}

; Check that the below loops are exchanged to allow innermost loop
; vectorization. We cannot vectorize the j-loop because it has a lexically
; backward dependency, but the i-loop can be vectorized because all the
; loop-carried dependencies are lexically forward. LoopVectorize currently only
; vectorizes innermost loop, hence move the i-loop to that position.
;
; for (int i = 0; i < 255; i++) {
;   for (int j = 1; j < 256; j++) {
;     A[i][j] = A[i][j-1] + B[i][j];
;     C[i][j] += C[i+1][j];
;   }
; }
;

; CHECK:      --- !Passed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        interchange_necessary_for_vectorization2
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Loop interchanged with enclosing loop.
define void @interchange_necessary_for_vectorization2() {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 1, %entry ], [ %i.next, %for.i.inc ]
  %i.inc = add nuw nsw i64 %i, 1
  br label %for.j.body

for.j.body:
  %j = phi i64 [ 1, %for.i.header ], [ %j.next, %for.j.body ]
  %j.dec = add nsw i64 %j, -1
  %a.load.index = getelementptr inbounds [256 x [256 x float]], ptr @A, i64 0, i64 %i, i64 %j.dec
  %b.index = getelementptr inbounds [256 x [256 x float]], ptr @B, i64 0, i64 %i, i64 %j
  %c.load.index = getelementptr inbounds [256 x [256 x float]], ptr @C, i64 0, i64 %i.inc, i64 %j
  %c.store.index = getelementptr inbounds [256 x [256 x float]], ptr @C, i64 0, i64 %i, i64 %j
  %a = load float, ptr %a.load.index
  %b = load float, ptr %b.index
  %c0 = load float, ptr %c.load.index
  %c1 = load float, ptr %c.store.index
  %add.0 = fadd float %a, %b
  %a.store.index = getelementptr inbounds [256 x [256 x float]], ptr @A, i64 0, i64 %i, i64 %j
  store float %add.0, ptr %a.store.index
  %add.1 = fadd float %c0, %c1
  store float %add.1, ptr %c.store.index
  %j.next = add nuw nsw i64 %j, 1
  %cmp.j = icmp eq i64 %j.next, 256
  br i1 %cmp.j, label %for.i.inc, label %for.j.body

for.i.inc:
  %i.next = add nuw nsw i64 %i, 1
  %cmp.i = icmp eq i64 %i.next, 255
  br i1 %cmp.i, label %exit, label %for.i.header

exit:
  ret void
}

; Check that no interchange is performed for the following loop. Interchanging
; the j-loop and k-loop makes the innermost loop vectorizble, since the j-loop
; has only forward dependencies. However, at the moment, a loop body consisting
; of multiple BBs is handled pesimistically. Hence the j-loop isn't moved to
; the innermost place.
;
; for (int i = 0; i < 255; i++) {
;   for (int j = 0; j < 255; j++) {
;     for (int k = 0; k < 128; k++) {
;       E[i][j][k] = D[i+1][j+1][2*k];
;       if (cond)
;         D[i][j][k+1] = 1.0;
;   }
; }

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            InterchangeNotProfitable
; CHECK-NEXT: Function:        multiple_BBs_in_loop
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Interchanging loops is not considered to improve cache locality nor vectorization.
; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            InterchangeNotProfitable
; CHECK-NEXT: Function:        multiple_BBs_in_loop
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Interchanging loops is not considered to improve cache locality nor vectorization.
define void @multiple_BBs_in_loop() {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %for.i.inc ]
  %i.inc = add nuw nsw i64 %i, 1
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.header ], [ %j.inc, %for.j.inc ]
  %j.inc = add nuw nsw i64 %j, 1
  br label %for.k.body

for.k.body:
  %k = phi i64 [ 0, %for.j.header ], [ %k.inc, %for.k.inc ]
  %k.inc = add nuw nsw i64 %k, 1
  %k.2 = mul nuw nsw i64 %k, 2
  %d.index = getelementptr inbounds [256 x [256 x [256 x float]]], ptr @D, i64 0, i64 %i.inc, i64 %j.inc, i64 %k.2
  %e.index = getelementptr inbounds [256 x [256 x [256 x float]]], ptr @E, i64 0, i64 %i, i64 %j, i64 %k
  %d.load = load float, ptr %d.index
  store float %d.load, ptr %e.index
  %cond = freeze i1 undef
  br i1 %cond, label %if.then, label %for.k.inc

if.then:
  %d.index2 = getelementptr inbounds [256 x [256 x [256 x float]]], ptr @D, i64 0, i64 %i, i64 %j, i64 %k.inc
  store float 1.0, ptr %d.index2
  br label %for.k.inc

for.k.inc:
  %cmp.k = icmp eq i64 %k.inc, 128
  br i1 %cmp.k, label %for.j.inc, label %for.k.body

for.j.inc:
  %cmp.j = icmp eq i64 %j.inc, 255
  br i1 %cmp.j, label %for.i.inc, label %for.j.header

for.i.inc:
  %cmp.i = icmp eq i64 %i.inc, 255
  br i1 %cmp.i, label %exit, label %for.i.header

exit:
  ret void
}
