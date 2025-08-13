; RUN: opt < %s -passes='print<delinearization>' -disable-output -delinearize-use-fixed-size-array-heuristic 2>&1 | FileCheck %s

; void f(int A[][8][32]) {
;   for (i = 0; i < 42; i++)
;    for (j = 0; j < 8; j++)
;     for (k = 0; k < 32; k++)
;       A[i][j][k] = 1;
; }

; CHECK:      Delinearization on function a_i_j_k:
; CHECK:      Base offset: %a
; CHECK-NEXT: ArrayDecl[UnknownSize][8][32] with elements of 4 bytes.
; CHECK-NEXT: ArrayRef[{0,+,1}<nuw><nsw><%for.i.header>][{0,+,1}<nuw><nsw><%for.j.header>][{0,+,1}<nuw><nsw><%for.k>]
define void @a_i_j_k(ptr %a) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  br label %for.j.header

for.j.header:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j.latch ]
  br label %for.k

for.k:
  %k = phi i32 [ 0, %for.j.header ], [ %k.inc, %for.k ]
  %idx = getelementptr [8 x [32 x i32]], ptr %a, i32 %i, i32 %j, i32 %k
  store i32 1, ptr %idx
  %k.inc = add i32 %k, 1
  %cmp.k = icmp slt i32 %k.inc, 32
  br i1 %cmp.k, label %for.k, label %for.j.latch

for.j.latch:
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 8
  br i1 %cmp.j, label %for.j.header, label %for.i.latch

for.i.latch:
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 42
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}

; void f(int A[][8][32]) {
;   for (i = 0; i < 42; i++)
;    for (j = 0; j < 8; j++)
;     for (k = 0; k < 32; k++)
;       A[i][7-j][k] = 1;
; }

; CHECK:      Delinearization on function a_i_nj_k:
; CHECK:      Base offset: %a
; CHECK-NEXT: ArrayDecl[UnknownSize][8][32] with elements of 4 bytes.
; CHECK-NEXT: ArrayRef[{0,+,1}<nuw><nsw><%for.i.header>][{7,+,-1}<nsw><%for.j.header>][{0,+,1}<nuw><nsw><%for.k>]
define void @a_i_nj_k(ptr %a) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  br label %for.j.header

for.j.header:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j.latch ]
  %j.subscript = sub i32 7, %j
  br label %for.k

for.k:
  %k = phi i32 [ 0, %for.j.header ], [ %k.inc, %for.k ]
  %idx = getelementptr [8 x [32 x i32]], ptr %a, i32 %i, i32 %j.subscript, i32 %k
  store i32 1, ptr %idx
  %k.inc = add i32 %k, 1
  %cmp.k = icmp slt i32 %k.inc, 32
  br i1 %cmp.k, label %for.k, label %for.j.latch

for.j.latch:
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 8
  br i1 %cmp.j, label %for.j.header, label %for.i.latch

for.i.latch:
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 42
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}

; In the following code, the access functions for both stores are represented
; in the same way in SCEV, so the delinearization results are also the same. We
; don't have any type information of the underlying objects.
;
; void f(int A[][4][64], int B[][8][32]) {
;   for (i = 0; i < 42; i++)
;    for (j = 0; j < 4; j++)
;     for (k = 0; k < 32; k++) {
;       A[i][j][k] = 1;
;       B[i][2*j][k] = 1;
;     }
; }

; CHECK:      Delinearization on function a_ijk_b_i2jk:
; CHECK:      Base offset: %a
; CHECK-NEXT: ArrayDecl[UnknownSize][4][64] with elements of 4 bytes.
; CHECK-NEXT: ArrayRef[{0,+,1}<nuw><nsw><%for.i.header>][{0,+,1}<nuw><nsw><%for.j.header>][{0,+,1}<nuw><nsw><%for.k>]
; CHECK:      Base offset: %b
; CHECK-NEXT: ArrayDecl[UnknownSize][4][64] with elements of 4 bytes.
; CHECK-NEXT: ArrayRef[{0,+,1}<nuw><nsw><%for.i.header>][{0,+,1}<nuw><nsw><%for.j.header>][{0,+,1}<nuw><nsw><%for.k>]
define void @a_ijk_b_i2jk(ptr %a, ptr %b) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  br label %for.j.header

for.j.header:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j.latch ]
  %j2 = shl i32 %j, 1
  br label %for.k

for.k:
  %k = phi i32 [ 0, %for.j.header ], [ %k.inc, %for.k ]
  %a.idx = getelementptr [4 x [64 x i32]], ptr %a, i32 %i, i32 %j, i32 %k
  %b.idx = getelementptr [8 x [32 x i32]], ptr %b, i32 %i, i32 %j2, i32 %k
  store i32 1, ptr %a.idx
  store i32 1, ptr %b.idx
  %k.inc = add i32 %k, 1
  %cmp.k = icmp slt i32 %k.inc, 32
  br i1 %cmp.k, label %for.k, label %for.j.latch

for.j.latch:
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 4
  br i1 %cmp.j, label %for.j.header, label %for.i.latch

for.i.latch:
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 42
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}

; The type information of the underlying object is not available, so the
; delinearization result is different from the original array size. In this
; case, the underlying object is a type of int[][8][32], but the
; delinearization result is like int[][4][64].
;
; void f(int A[][8][32]) {
;   for (i = 0; i < 42; i++)
;    for (j = 0; j < 3; j++)
;     for (k = 0; k < 32; k++)
;       A[i][2*j+1][k] = 1;
; }

; CHECK:      Delinearization on function a_i_2j1_k:
; CHECK:      Base offset: %a
; CHECK-NEXT: ArrayDecl[UnknownSize][4][64] with elements of 4 bytes.
; CHECK-NEXT: ArrayRef[{0,+,1}<nuw><nsw><%for.i.header>][{0,+,1}<nuw><%for.j.header>][{32,+,1}<nw><%for.k>]
define void @a_i_2j1_k(ptr %a) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  br label %for.j.header

for.j.header:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j.latch ]
  %j2 = shl i32 %j, 1
  %j.subscript = add i32 %j2, 1
  br label %for.k

for.k:
  %k = phi i32 [ 0, %for.j.header ], [ %k.inc, %for.k ]
  %idx = getelementptr [8 x [32 x i32]], ptr %a, i32 %i, i32 %j.subscript, i32 %k
  store i32 1, ptr %idx
  %k.inc = add i32 %k, 1
  %cmp.k = icmp slt i32 %k.inc, 32
  br i1 %cmp.k, label %for.k, label %for.j.latch

for.j.latch:
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 3
  br i1 %cmp.j, label %for.j.header, label %for.i.latch

for.i.latch:
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 42
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}

; Fail to delinearize because the step recurrence value of the i-loop is not
; divisible by that of the j-loop.
;
; void f(int A[][8][32]) {
;   for (i = 0; i < 42; i++)
;    for (j = 0; j < 2; j++)
;     for (k = 0; k < 42; k++)
;       A[i][3*j][k] = 1;
; }

; CHECK:      Delinearization on function a_i_3j_k:
; CHECK:      AccessFunction: {{...}}0,+,1024}<nuw><nsw><%for.i.header>,+,384}<nw><%for.j.header>,+,4}<nw><%for.k>
; CHECK-NEXT: failed to delinearize
define void @a_i_3j_k(ptr %a) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  br label %for.j.header

for.j.header:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j.latch ]
  %j.subscript = mul i32 %j, 3
  br label %for.k

for.k:
  %k = phi i32 [ 0, %for.j.header ], [ %k.inc, %for.k ]
  %idx = getelementptr [8 x [32 x i32]], ptr %a, i32 %i, i32 %j.subscript, i32 %k
  store i32 1, ptr %idx
  %k.inc = add i32 %k, 1
  %cmp.k = icmp slt i32 %k.inc, 42
  br i1 %cmp.k, label %for.k, label %for.j.latch

for.j.latch:
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 2
  br i1 %cmp.j, label %for.j.header, label %for.i.latch

for.i.latch:
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 42
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}

; Although the step recurrence value of j-loop is not divisible by that of the
; k-loop, delinearization is possible because we know that the "actual" stride
; width for the last dimension is 4 instead of 12.
;
; void f(int A[][8][32]) {
;   for (i = 0; i < 42; i++)
;    for (j = 0; j < 8; j++)
;     for (k = 0; k < 10; k++)
;       A[i][j][3*k] = 1;
; }

; CHECK:      Delinearization on function a_i_j_3k:
; CHECK:      Base offset: %a
; CHECK-NEXT: ArrayDecl[UnknownSize][8][32] with elements of 4 bytes.
; CHECK-NEXT: ArrayRef[{0,+,1}<nuw><nsw><%for.i.header>][{0,+,1}<nuw><nsw><%for.j.header>][{0,+,3}<nuw><nsw><%for.k>]
define void @a_i_j_3k(ptr %a) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  br label %for.j.header

for.j.header:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j.latch ]
  br label %for.k

for.k:
  %k = phi i32 [ 0, %for.j.header ], [ %k.inc, %for.k ]
  %k.subscript = mul i32 %k, 3
  %idx = getelementptr [8 x [32 x i32]], ptr %a, i32 %i, i32 %j, i32 %k.subscript
  store i32 1, ptr %idx
  %k.inc = add i32 %k, 1
  %cmp.k = icmp slt i32 %k.inc, 10
  br i1 %cmp.k, label %for.k, label %for.j.latch

for.j.latch:
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 8
  br i1 %cmp.j, label %for.j.header, label %for.i.latch

for.i.latch:
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 42
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}

; Fail to delinearize because i is used in multiple subscripts that are not adjacent.
;
; void f(int A[][8][32]) {
;   for (i = 0; i < 32; i++)
;    for (j = 0; j < 2; j++)
;     for (k = 0; k < 4; k++)
;       A[i][2*j+k][i] = 1;
; }

; CHECK:      Delinearization on function a_i_j2k_i:
; CHECK:      AccessFunction: {{...}}0,+,1028}<%for.i.header>,+,256}<nw><%for.j.header>,+,128}<nw><%for.k>
; CHECK-NEXT: failed to delinearize
define void @a_i_j2k_i(ptr %a) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  br label %for.j.header

for.j.header:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j.latch ]
  br label %for.k

for.k:
  %k = phi i32 [ 0, %for.j.header ], [ %k.inc, %for.k ]
  %j2 = shl i32 %j, 1
  %j2.k = add i32 %j2, %k
  %idx = getelementptr [8 x [32 x i32]], ptr %a, i32 %i, i32 %j2.k, i32 %i
  store i32 1, ptr %idx
  %k.inc = add i32 %k, 1
  %cmp.k = icmp slt i32 %k.inc, 4
  br i1 %cmp.k, label %for.k, label %for.j.latch

for.j.latch:
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 2
  br i1 %cmp.j, label %for.j.header, label %for.i.latch

for.i.latch:
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 32
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}

; Can delinearize, but the result is different from the original array size. In
; this case, the outermost two dimensions are melded into one.
;
; void f(int A[][8][32]) {
;   for (i = 0; i < 8; i++)
;    for (j = 0; j < 10; j++)
;     for (k = 0; k < 10; k++)
;       A[i][i][j+k] = 1;
; }

; CHECK:      Delinearization on function a_i_i_jk:
; CHECK:      Base offset: %a
; CHECK-NEXT: ArrayDecl[UnknownSize][288] with elements of 4 bytes.
; CHECK-NEXT: ArrayRef[{0,+,1}<nuw><nsw><%for.i.header>][{{..}}0,+,1}<nuw><nsw><%for.j.header>,+,1}<nuw><nsw><%for.k>]
define void @a_i_i_jk(ptr %a) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  br label %for.j.header

for.j.header:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j.latch ]
  br label %for.k

for.k:
  %k = phi i32 [ 0, %for.j.header ], [ %k.inc, %for.k ]
  %jk = add i32 %j, %k
  %idx = getelementptr [8 x [32 x i32]], ptr %a, i32 %i, i32 %i, i32 %jk
  store i32 1, ptr %idx
  %k.inc = add i32 %k, 1
  %cmp.k = icmp slt i32 %k.inc, 10
  br i1 %cmp.k, label %for.k, label %for.j.latch

for.j.latch:
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 10
  br i1 %cmp.j, label %for.j.header, label %for.i.latch

for.i.latch:
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 8
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}

; void f(int A[][8][32]) {
;   for (i = 0; i < 8; i++)
;    for (j = 0; j < 4; j++)
;     for (k = 0; k < 4; k++)
;       for (l = 0; l < 32; l++)
;         A[i][j+k][l] = 1;
; }

; CHECK:      Delinearization on function a_i_jk_l:
; CHECK:      Base offset: %a
; CHECK-NEXT: ArrayDecl[UnknownSize][8][32] with elements of 4 bytes.
; CHECK-NEXT: ArrayRef[{0,+,1}<nuw><nsw><%for.i.header>][{{..}}0,+,1}<nuw><nsw><%for.j.header>,+,1}<nuw><nsw><%for.k.header>][{0,+,1}<nuw><nsw><%for.l>]

define void @a_i_jk_l(ptr %a) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  br label %for.j.header

for.j.header:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j.latch ]
  br label %for.k.header

for.k.header:
  %k = phi i32 [ 0, %for.j.header ], [ %k.inc, %for.k.latch ]
  %jk = add i32 %j, %k
  br label %for.l

for.l:
  %l = phi i32 [ 0, %for.k.header ], [ %l.inc, %for.l ]
  %idx = getelementptr [8 x [32 x i32]], ptr %a, i32 %i, i32 %jk, i32 %l
  store i32 1, ptr %idx
  %l.inc = add i32 %l, 1
  %cmp.l = icmp slt i32 %l.inc, 32
  br i1 %cmp.l, label %for.l, label %for.k.latch

for.k.latch:
  %k.inc = add i32 %k, 1
  %cmp.k = icmp slt i32 %k.inc, 4
  br i1 %cmp.k, label %for.k.header, label %for.j.latch

for.j.latch:
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 4
  br i1 %cmp.j, label %for.j.header, label %for.i.latch

for.i.latch:
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 8
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}

; Reject if the address is not a multiple of the element size.
;
; void f(int *A) {
;   for (i = 0; i < 42; i++)
;    for (j = 0; j < 8; j++)
;     for (k = 0; k < 32; k++)
;       *((int *)((char *)A + i*256 + j*32 + k)) = 1;
; }

; CHECK:      Delinearization on function non_divisible_by_element_size:
; CHECK:      AccessFunction: {{...}}0,+,256}<nuw><nsw><%for.i.header>,+,32}<nw><%for.j.header>,+,1}<nw><%for.k>
; CHECK-NEXT: failed to delinearize
define void @non_divisible_by_element_size(ptr %a) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  br label %for.j.header

for.j.header:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j.latch ]
  br label %for.k

for.k:
  %k = phi i32 [ 0, %for.j.header ], [ %k.inc, %for.k ]
  %idx = getelementptr [8 x [32 x i8]], ptr %a, i32 %i, i32 %j, i32 %k
  store i32 1, ptr %idx
  %k.inc = add i32 %k, 1
  %cmp.k = icmp slt i32 %k.inc, 32
  br i1 %cmp.k, label %for.k, label %for.j.latch

for.j.latch:
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 8
  br i1 %cmp.j, label %for.j.header, label %for.i.latch

for.i.latch:
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 42
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}
