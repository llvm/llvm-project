; RUN: opt %loadPolly -polly-print-scops -polly-ignore-aliasing -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-print-function-scops -polly-ignore-aliasing -disable-output < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; typedef struct Mat {
;   float* data;
;   long rows;
;   long cols;
;   long deps;
; } Matrix;
;
; #define M(m,r,c,d)  m->data[(r) * m->cols * mt->mdeps + (c) * m->deps + (d)]
;
; void jacobi(long nn, Matrix* a1, ..., Matrix* a19) {
;   long i, j, k, n, imax, jmax, kmax;
;
;   for(n = 0 ; n < nn ; n++)
;     for(i = 1 ; i < a1->rows-1; i++)
;       for(j = 1 ; j < a1->cols-1; j++)
;         for(k = 1 ; k < a1->deps-1; k++)
;           M(a19,i,j,k) = M(a1,i,j,k) + ... + M(a18,i,j,k);
; }
;
; We applied some loop invariant code motion to ensure the loads from m->data,
; m->rows, m->cols and m->deps happen before the scop.

; This test case verifies that the construction of the assumed context finishes
; successfully. Depending on how constrained are accumulated in the assumed
; context, this test case can take even for a smaller number of arrays over a
; minute to complete. With the unrolling choosen in this test, an inefficient
; formulation of the assumption tracking cause LLVM to crash due to excessive
; memory usage due to an overly large number of disjuncts being formed.

%struct.Mat = type { float*, i64, i64, i64 }

; CHECK: Assumed Context:
; CHECK-DAG: a2_cols >= -1 + a1_cols
; CHECK-DAG: a2_deps >= -1 + a1_deps
; CHECK-DAG: a3_cols >= -1 + a1_cols
; CHECK-DAG: a3_deps >= -1 + a1_deps
; CHECK-DAG: a4_cols >= -1 + a1_cols
; CHECK-DAG: a4_deps >= -1 + a1_deps
; CHECK-DAG: a5_cols >= -1 + a1_cols
; CHECK-DAG: a5_deps >= -1 + a1_deps
; CHECK-DAG: a6_cols >= -1 + a1_cols
; CHECK-DAG: a6_deps >= -1 + a1_deps
; CHECK-DAG: a7_cols >= -1 + a1_cols
; CHECK-DAG: a7_deps >= -1 + a1_deps
; CHECK-DAG: a8_cols >= -1 + a1_cols
; CHECK-DAG: a8_deps >= -1 + a1_deps
; CHECK-DAG: a9_deps >= -1 + a1_deps
; CHECK-DAG: a9_cols >= -1 + a1_cols
; CHECK-DAG: a10_cols >= -1 + a1_cols
; CHECK-DAG: a10_deps >= -1 + a1_deps
; CHECK-DAG: a11_cols >= -1 + a1_cols
; CHECK-DAG: a11_deps >= -1 + a1_deps
; CHECK-DAG: a12_cols >= -1 + a1_cols
; CHECK-DAG: a12_deps >= -1 + a1_deps
; CHECK-DAG: a13_cols >= -1 + a1_cols
; CHECK-DAG: a13_deps >= -1 + a1_deps
; CHECK-DAG: a14_cols >= -1 + a1_cols
; CHECK-DAG: a14_deps >= -1 + a1_deps
; CHECK-DAG: a15_cols >= -1 + a1_cols
; CHECK-DAG: a15_deps >= -1 + a1_deps
; CHECK-DAG: a16_cols >= -1 + a1_cols
; CHECK-DAG: a16_deps >= -1 + a1_deps
; CHECK-DAG: a17_cols >= -1 + a1_cols
; CHECK-DAG: a17_deps >= -1 + a1_deps
; CHECK-DAG: a18_cols >= -1 + a1_cols
; CHECK-DAG: a18_deps >= -1 + a1_deps
; CHECK-DAG: a19_deps >= -1 + a1_deps
; CHECK-DAG: a19_cols >= -1 + a1_cols

define void @jacobi(i64 %nn,
  %struct.Mat* %a1,
  %struct.Mat* %a2,
  %struct.Mat* %a3,
  %struct.Mat* %a4,
  %struct.Mat* %a5,
  %struct.Mat* %a6,
  %struct.Mat* %a7,
  %struct.Mat* %a8,
  %struct.Mat* %a9,
  %struct.Mat* %a10,
  %struct.Mat* %a11,
  %struct.Mat* %a12,
  %struct.Mat* %a13,
  %struct.Mat* %a14,
  %struct.Mat* %a15,
  %struct.Mat* %a16,
  %struct.Mat* %a17,
  %struct.Mat* %a18,
  %struct.Mat* %a19
  )  {

entry:
  br label %entry.split

entry.split:
  %a1.data.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a1, i32 0, i32 0
  %a1.data = load float*, float** %a1.data.ptr, align 8
  %a1.rows.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a1, i32 0, i32 1
  %a1.rows = load i64, i64* %a1.rows.ptr, align 8
  %a1.cols.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a1, i32 0, i32 2
  %a1.cols = load i64, i64* %a1.cols.ptr, align 8
  %a1.deps.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a1, i32 0, i32 3
  %a1.deps = load i64, i64* %a1.deps.ptr, align 8

  %a2.data.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a2, i32 0, i32 0
  %a2.data = load float*, float** %a2.data.ptr, align 8
  %a2.cols.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a2, i32 0, i32 2
  %a2.cols = load i64, i64* %a2.cols.ptr, align 8
  %a2.deps.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a2, i32 0, i32 3
  %a2.deps = load i64, i64* %a2.deps.ptr, align 8

  %a3.data.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a3, i32 0, i32 0
  %a3.data = load float*, float** %a3.data.ptr, align 8
  %a3.cols.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a3, i32 0, i32 2
  %a3.cols = load i64, i64* %a3.cols.ptr, align 8
  %a3.deps.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a3, i32 0, i32 3
  %a3.deps = load i64, i64* %a3.deps.ptr, align 8

  %a4.data.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a4, i32 0, i32 0
  %a4.data = load float*, float** %a4.data.ptr, align 8
  %a4.cols.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a4, i32 0, i32 2
  %a4.cols = load i64, i64* %a4.cols.ptr, align 8
  %a4.deps.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a4, i32 0, i32 3
  %a4.deps = load i64, i64* %a4.deps.ptr, align 8

  %a5.data.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a5, i32 0, i32 0
  %a5.data = load float*, float** %a5.data.ptr, align 8
  %a5.cols.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a5, i32 0, i32 2
  %a5.cols = load i64, i64* %a5.cols.ptr, align 8
  %a5.deps.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a5, i32 0, i32 3
  %a5.deps = load i64, i64* %a5.deps.ptr, align 8

  %a6.data.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a6, i32 0, i32 0
  %a6.data = load float*, float** %a6.data.ptr, align 8
  %a6.cols.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a6, i32 0, i32 2
  %a6.cols = load i64, i64* %a6.cols.ptr, align 8
  %a6.deps.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a6, i32 0, i32 3
  %a6.deps = load i64, i64* %a6.deps.ptr, align 8

  %a7.data.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a7, i32 0, i32 0
  %a7.data = load float*, float** %a7.data.ptr, align 8
  %a7.cols.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a7, i32 0, i32 2
  %a7.cols = load i64, i64* %a7.cols.ptr, align 8
  %a7.deps.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a7, i32 0, i32 3
  %a7.deps = load i64, i64* %a7.deps.ptr, align 8

  %a8.data.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a8, i32 0, i32 0
  %a8.data = load float*, float** %a8.data.ptr, align 8
  %a8.cols.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a8, i32 0, i32 2
  %a8.cols = load i64, i64* %a8.cols.ptr, align 8
  %a8.deps.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a8, i32 0, i32 3
  %a8.deps = load i64, i64* %a8.deps.ptr, align 8

  %a9.data.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a9, i32 0, i32 0
  %a9.data = load float*, float** %a9.data.ptr, align 8
  %a9.cols.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a9, i32 0, i32 2
  %a9.cols = load i64, i64* %a9.cols.ptr, align 8
  %a9.deps.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a9, i32 0, i32 3
  %a9.deps = load i64, i64* %a9.deps.ptr, align 8

  %a10.data.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a10, i32 0, i32 0
  %a10.data = load float*, float** %a10.data.ptr, align 8
  %a10.cols.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a10, i32 0, i32 2
  %a10.cols = load i64, i64* %a10.cols.ptr, align 8
  %a10.deps.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a10, i32 0, i32 3
  %a10.deps = load i64, i64* %a10.deps.ptr, align 8

  %a11.data.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a11, i32 0, i32 0
  %a11.data = load float*, float** %a11.data.ptr, align 8
  %a11.cols.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a11, i32 0, i32 2
  %a11.cols = load i64, i64* %a11.cols.ptr, align 8
  %a11.deps.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a11, i32 0, i32 3
  %a11.deps = load i64, i64* %a11.deps.ptr, align 8

  %a12.data.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a12, i32 0, i32 0
  %a12.data = load float*, float** %a12.data.ptr, align 8
  %a12.cols.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a12, i32 0, i32 2
  %a12.cols = load i64, i64* %a12.cols.ptr, align 8
  %a12.deps.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a12, i32 0, i32 3
  %a12.deps = load i64, i64* %a12.deps.ptr, align 8

  %a13.data.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a13, i32 0, i32 0
  %a13.data = load float*, float** %a13.data.ptr, align 8
  %a13.cols.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a13, i32 0, i32 2
  %a13.cols = load i64, i64* %a13.cols.ptr, align 8
  %a13.deps.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a13, i32 0, i32 3
  %a13.deps = load i64, i64* %a13.deps.ptr, align 8

  %a14.data.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a14, i32 0, i32 0
  %a14.data = load float*, float** %a14.data.ptr, align 8
  %a14.cols.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a14, i32 0, i32 2
  %a14.cols = load i64, i64* %a14.cols.ptr, align 8
  %a14.deps.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a14, i32 0, i32 3
  %a14.deps = load i64, i64* %a14.deps.ptr, align 8

  %a15.data.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a15, i32 0, i32 0
  %a15.data = load float*, float** %a15.data.ptr, align 8
  %a15.cols.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a15, i32 0, i32 2
  %a15.cols = load i64, i64* %a15.cols.ptr, align 8
  %a15.deps.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a15, i32 0, i32 3
  %a15.deps = load i64, i64* %a15.deps.ptr, align 8

  %a16.data.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a16, i32 0, i32 0
  %a16.data = load float*, float** %a16.data.ptr, align 8
  %a16.cols.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a16, i32 0, i32 2
  %a16.cols = load i64, i64* %a16.cols.ptr, align 8
  %a16.deps.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a16, i32 0, i32 3
  %a16.deps = load i64, i64* %a16.deps.ptr, align 8

  %a17.data.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a17, i32 0, i32 0
  %a17.data = load float*, float** %a17.data.ptr, align 8
  %a17.cols.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a17, i32 0, i32 2
  %a17.cols = load i64, i64* %a17.cols.ptr, align 8
  %a17.deps.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a17, i32 0, i32 3
  %a17.deps = load i64, i64* %a17.deps.ptr, align 8

  %a18.data.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a18, i32 0, i32 0
  %a18.data = load float*, float** %a18.data.ptr, align 8
  %a18.cols.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a18, i32 0, i32 2
  %a18.cols = load i64, i64* %a18.cols.ptr, align 8
  %a18.deps.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a18, i32 0, i32 3
  %a18.deps = load i64, i64* %a18.deps.ptr, align 8

  %a19.data.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a19, i32 0, i32 0
  %a19.data = load float*, float** %a19.data.ptr, align 8
  %a19.cols.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a19, i32 0, i32 2
  %a19.cols = load i64, i64* %a19.cols.ptr, align 8
  %a19.deps.ptr = getelementptr inbounds %struct.Mat, %struct.Mat* %a19, i32 0, i32 3
  %a19.deps = load i64, i64* %a19.deps.ptr, align 8
  br label %for.n

for.n:
  %indvar.n = phi i64 [ 0, %entry.split ], [ %indvar.n.next, %for.inc.n ]
  br label %for.i

for.i:
  %indvar.i = phi i64 [ 0, %for.n ], [ %indvar.i.next, %for.inc.i ]
  br label %for.j

for.j:
  %indvar.j = phi i64 [ 0, %for.i ], [ %indvar.j.next, %for.inc.j ]
  br label %for.body.k

for.body.k:
  %indvar.k = phi i64 [ 0, %for.j ], [ %indvar.k.next, %for.body.k ]
  %a1.tmp1 = add i64 %a1.cols, 1
  %a1.tmp2 = mul i64 %a1.deps, %a1.tmp1
  %a1.tmp3 = add i64 %a1.tmp2, 1
  %a1.tmp4 = mul i64 %a1.cols, %a1.deps
  %a1.idx.i = mul i64 %a1.tmp4, %indvar.i
  %a1.tmp5 = add i64 %a1.tmp3, %a1.idx.i
  %a1.idx.j = mul i64 %a1.deps, %indvar.j
  %a1.tmp6 = add i64 %a1.tmp5, %a1.idx.j
  %a1.idx.k = add i64 %a1.tmp6, %indvar.k
  %a1.ptr = getelementptr float, float* %a1.data, i64 %a1.idx.k
  %sum = load float, float* %a1.ptr, align 4

  %a2.tmp1 = add i64 %a2.cols, 1
  %a2.tmp2 = mul i64 %a2.deps, %a2.tmp1
  %a2.tmp3 = add i64 %a2.tmp2, 1
  %a2.tmp4 = mul i64 %a2.cols, %a2.deps
  %a2.idx.i = mul i64 %a2.tmp4, %indvar.i
  %a2.tmp5 = add i64 %a2.tmp3, %a2.idx.i
  %a2.idx.j = mul i64 %a2.deps, %indvar.j
  %a2.tmp6 = add i64 %a2.tmp5, %a2.idx.j
  %a2.idx.k = add i64 %a2.tmp6, %indvar.k
  %a2.ptr = getelementptr float, float* %a2.data, i64 %a2.idx.k
  %a2.val = load float, float* %a2.ptr, align 4
  %sum.a2 = fadd float %sum, %a2.val

  %a3.tmp1 = add i64 %a3.cols, 1
  %a3.tmp2 = mul i64 %a3.deps, %a3.tmp1
  %a3.tmp3 = add i64 %a3.tmp2, 1
  %a3.tmp4 = mul i64 %a3.cols, %a3.deps
  %a3.idx.i = mul i64 %a3.tmp4, %indvar.i
  %a3.tmp5 = add i64 %a3.tmp3, %a3.idx.i
  %a3.idx.j = mul i64 %a3.deps, %indvar.j
  %a3.tmp6 = add i64 %a3.tmp5, %a3.idx.j
  %a3.idx.k = add i64 %a3.tmp6, %indvar.k
  %a3.ptr = getelementptr float, float* %a3.data, i64 %a3.idx.k
  %a3.val = load float, float* %a3.ptr, align 4
  %sum.a3 = fadd float %sum.a2, %a3.val

  %a4.tmp1 = add i64 %a4.cols, 1
  %a4.tmp2 = mul i64 %a4.deps, %a4.tmp1
  %a4.tmp3 = add i64 %a4.tmp2, 1
  %a4.tmp4 = mul i64 %a4.cols, %a4.deps
  %a4.idx.i = mul i64 %a4.tmp4, %indvar.i
  %a4.tmp5 = add i64 %a4.tmp3, %a4.idx.i
  %a4.idx.j = mul i64 %a4.deps, %indvar.j
  %a4.tmp6 = add i64 %a4.tmp5, %a4.idx.j
  %a4.idx.k = add i64 %a4.tmp6, %indvar.k
  %a4.ptr = getelementptr float, float* %a4.data, i64 %a4.idx.k
  %a4.val = load float, float* %a4.ptr, align 4
  %sum.a4 = fadd float %sum.a3, %a4.val

  %a5.tmp1 = add i64 %a5.cols, 1
  %a5.tmp2 = mul i64 %a5.deps, %a5.tmp1
  %a5.tmp3 = add i64 %a5.tmp2, 1
  %a5.tmp4 = mul i64 %a5.cols, %a5.deps
  %a5.idx.i = mul i64 %a5.tmp4, %indvar.i
  %a5.tmp5 = add i64 %a5.tmp3, %a5.idx.i
  %a5.idx.j = mul i64 %a5.deps, %indvar.j
  %a5.tmp6 = add i64 %a5.tmp5, %a5.idx.j
  %a5.idx.k = add i64 %a5.tmp6, %indvar.k
  %a5.ptr = getelementptr float, float* %a5.data, i64 %a5.idx.k
  %a5.val = load float, float* %a5.ptr, align 4
  %sum.a5 = fadd float %sum.a4, %a5.val

  %a6.tmp1 = add i64 %a6.cols, 1
  %a6.tmp2 = mul i64 %a6.deps, %a6.tmp1
  %a6.tmp3 = add i64 %a6.tmp2, 1
  %a6.tmp4 = mul i64 %a6.cols, %a6.deps
  %a6.idx.i = mul i64 %a6.tmp4, %indvar.i
  %a6.tmp5 = add i64 %a6.tmp3, %a6.idx.i
  %a6.idx.j = mul i64 %a6.deps, %indvar.j
  %a6.tmp6 = add i64 %a6.tmp5, %a6.idx.j
  %a6.idx.k = add i64 %a6.tmp6, %indvar.k
  %a6.ptr = getelementptr float, float* %a6.data, i64 %a6.idx.k
  %a6.val = load float, float* %a6.ptr, align 4
  %sum.a6 = fadd float %sum.a5, %a6.val

  %a7.tmp1 = add i64 %a7.cols, 1
  %a7.tmp2 = mul i64 %a7.deps, %a7.tmp1
  %a7.tmp3 = add i64 %a7.tmp2, 1
  %a7.tmp4 = mul i64 %a7.cols, %a7.deps
  %a7.idx.i = mul i64 %a7.tmp4, %indvar.i
  %a7.tmp5 = add i64 %a7.tmp3, %a7.idx.i
  %a7.idx.j = mul i64 %a7.deps, %indvar.j
  %a7.tmp6 = add i64 %a7.tmp5, %a7.idx.j
  %a7.idx.k = add i64 %a7.tmp6, %indvar.k
  %a7.ptr = getelementptr float, float* %a7.data, i64 %a7.idx.k
  %a7.val = load float, float* %a7.ptr, align 4
  %sum.a7 = fadd float %sum.a6, %a7.val

  %a8.tmp1 = add i64 %a8.cols, 1
  %a8.tmp2 = mul i64 %a8.deps, %a8.tmp1
  %a8.tmp3 = add i64 %a8.tmp2, 1
  %a8.tmp4 = mul i64 %a8.cols, %a8.deps
  %a8.idx.i = mul i64 %a8.tmp4, %indvar.i
  %a8.tmp5 = add i64 %a8.tmp3, %a8.idx.i
  %a8.idx.j = mul i64 %a8.deps, %indvar.j
  %a8.tmp6 = add i64 %a8.tmp5, %a8.idx.j
  %a8.idx.k = add i64 %a8.tmp6, %indvar.k
  %a8.ptr = getelementptr float, float* %a8.data, i64 %a8.idx.k
  %a8.val = load float, float* %a8.ptr, align 4
  %sum.a8 = fadd float %sum.a7, %a8.val

  %a9.tmp1 = add i64 %a9.cols, 1
  %a9.tmp2 = mul i64 %a9.deps, %a9.tmp1
  %a9.tmp3 = add i64 %a9.tmp2, 1
  %a9.tmp4 = mul i64 %a9.cols, %a9.deps
  %a9.idx.i = mul i64 %a9.tmp4, %indvar.i
  %a9.tmp5 = add i64 %a9.tmp3, %a9.idx.i
  %a9.idx.j = mul i64 %a9.deps, %indvar.j
  %a9.tmp6  = add i64 %a9.tmp5, %a9.idx.j
  %a9.idx.k = add i64 %a9.tmp6, %indvar.k
  %a9.ptr = getelementptr float, float* %a9.data, i64 %a9.idx.k
  %a9.val = load float, float* %a9.ptr, align 4
  %sum.a9 = fadd float %sum.a8, %a9.val

  %a10.tmp1 = add i64 %a10.cols, 1
  %a10.tmp2 = mul i64 %a10.deps, %a10.tmp1
  %a10.tmp3 = add i64 %a10.tmp2, 1
  %a10.tmp4 = mul i64 %a10.cols, %a10.deps
  %a10.idx.i = mul i64 %a10.tmp4, %indvar.i
  %a10.tmp5 = add i64 %a10.tmp3, %a10.idx.i
  %a10.idx.j = mul i64 %a10.deps, %indvar.j
  %a10.tmp6 = add i64 %a10.tmp5, %a10.idx.j
  %a10.idx.k = add i64 %a10.tmp6, %indvar.k
  %a10.ptr = getelementptr float, float* %a10.data, i64 %a10.idx.k
  %a10.val = load float, float* %a10.ptr, align 4
  %sum.a10 = fadd float %sum.a9, %a10.val

  %a11.tmp1 = add i64 %a11.cols, 1
  %a11.tmp2 = mul i64 %a11.deps, %a11.tmp1
  %a11.tmp3 = add i64 %a11.tmp2, 1
  %a11.tmp4 = mul i64 %a11.cols, %a11.deps
  %a11.idx.i = mul i64 %a11.tmp4, %indvar.i
  %a11.tmp5 = add i64 %a11.tmp3, %a11.idx.i
  %a11.idx.j = mul i64 %a11.deps, %indvar.j
  %a11.tmp6 = add i64 %a11.tmp5, %a11.idx.j
  %a11.idx.k = add i64 %a11.tmp6, %indvar.k
  %a11.ptr = getelementptr float, float* %a11.data, i64 %a11.idx.k
  %a11.val = load float, float* %a11.ptr, align 4
  %sum.a11 = fadd float %sum.a10, %a11.val

  %a12.tmp1 = add i64 %a12.cols, 1
  %a12.tmp2 = mul i64 %a12.deps, %a12.tmp1
  %a12.tmp3 = add i64 %a12.tmp2, 1
  %a12.tmp4 = mul i64 %a12.cols, %a12.deps
  %a12.idx.i = mul i64 %a12.tmp4, %indvar.i
  %a12.tmp5 = add i64 %a12.tmp3, %a12.idx.i
  %a12.idx.j = mul i64 %a12.deps, %indvar.j
  %a12.tmp6 = add i64 %a12.tmp5, %a12.idx.j
  %a12.idx.k = add i64 %a12.tmp6, %indvar.k
  %a12.ptr = getelementptr float, float* %a12.data, i64 %a12.idx.k
  %a12.val = load float, float* %a12.ptr, align 4
  %sum.a12 = fadd float %sum.a11, %a12.val

  %a13.tmp1 = add i64 %a13.cols, 1
  %a13.tmp2 = mul i64 %a13.deps, %a13.tmp1
  %a13.tmp3 = add i64 %a13.tmp2, 1
  %a13.tmp4 = mul i64 %a13.cols, %a13.deps
  %a13.idx.i = mul i64 %a13.tmp4, %indvar.i
  %a13.tmp5 = add i64 %a13.tmp3, %a13.idx.i
  %a13.idx.j = mul i64 %a13.deps, %indvar.j
  %a13.tmp6 = add i64 %a13.tmp5, %a13.idx.j
  %a13.idx.k = add i64 %a13.tmp6, %indvar.k
  %a13.ptr = getelementptr float, float* %a13.data, i64 %a13.idx.k
  %a13.val = load float, float* %a13.ptr, align 4
  %sum.a13 = fadd float %sum.a12, %a13.val

  %a14.tmp1 = add i64 %a14.cols, 1
  %a14.tmp2 = mul i64 %a14.deps, %a14.tmp1
  %a14.tmp3 = add i64 %a14.tmp2, 1
  %a14.tmp4 = mul i64 %a14.cols, %a14.deps
  %a14.idx.i = mul i64 %a14.tmp4, %indvar.i
  %a14.tmp5 = add i64 %a14.tmp3, %a14.idx.i
  %a14.idx.j = mul i64 %a14.deps, %indvar.j
  %a14.tmp6 = add i64 %a14.tmp5, %a14.idx.j
  %a14.idx.k = add i64 %a14.tmp6, %indvar.k
  %a14.ptr = getelementptr float, float* %a14.data, i64 %a14.idx.k
  %a14.val = load float, float* %a14.ptr, align 4
  %sum.a14 = fadd float %sum.a13, %a14.val

  %a15.tmp1 = add i64 %a15.cols, 1
  %a15.tmp2 = mul i64 %a15.deps, %a15.tmp1
  %a15.tmp3 = add i64 %a15.tmp2, 1
  %a15.tmp4 = mul i64 %a15.cols, %a15.deps
  %a15.idx.i = mul i64 %a15.tmp4, %indvar.i
  %a15.tmp5 = add i64 %a15.tmp3, %a15.idx.i
  %a15.idx.j = mul i64 %a15.deps, %indvar.j
  %a15.tmp6 = add i64 %a15.tmp5, %a15.idx.j
  %a15.idx.k = add i64 %a15.tmp6, %indvar.k
  %a15.ptr = getelementptr float, float* %a15.data, i64 %a15.idx.k
  %a15.val = load float, float* %a15.ptr, align 4
  %sum.a15 = fadd float %sum.a14, %a15.val

  %a16.tmp1 = add i64 %a16.cols, 1
  %a16.tmp2 = mul i64 %a16.deps, %a16.tmp1
  %a16.tmp3 = add i64 %a16.tmp2, 1
  %a16.tmp4 = mul i64 %a16.cols, %a16.deps
  %a16.idx.i = mul i64 %a16.tmp4, %indvar.i
  %a16.tmp5 = add i64 %a16.tmp3, %a16.idx.i
  %a16.idx.j = mul i64 %a16.deps, %indvar.j
  %a16.tmp6 = add i64 %a16.tmp5, %a16.idx.j
  %a16.idx.k = add i64 %a16.tmp6, %indvar.k
  %a16.ptr = getelementptr float, float* %a16.data, i64 %a16.idx.k
  %a16.val = load float, float* %a16.ptr, align 4
  %sum.a16 = fadd float %sum.a15, %a16.val

  %a17.tmp1 = add i64 %a17.cols, 1
  %a17.tmp2 = mul i64 %a17.deps, %a17.tmp1
  %a17.tmp3 = add i64 %a17.tmp2, 1
  %a17.tmp4 = mul i64 %a17.cols, %a17.deps
  %a17.idx.i = mul i64 %a17.tmp4, %indvar.i
  %a17.tmp5 = add i64 %a17.tmp3, %a17.idx.i
  %a17.idx.j = mul i64 %a17.deps, %indvar.j
  %a17.tmp6 = add i64 %a17.tmp5, %a17.idx.j
  %a17.idx.k = add i64 %a17.tmp6, %indvar.k
  %a17.ptr = getelementptr float, float* %a17.data, i64 %a17.idx.k
  %a17.val = load float, float* %a17.ptr, align 4
  %sum.a17 = fadd float %sum.a16, %a17.val

  %a18.tmp1 = add i64 %a18.cols, 1
  %a18.tmp2 = mul i64 %a18.deps, %a18.tmp1
  %a18.tmp3 = add i64 %a18.tmp2, 1
  %a18.tmp4 = mul i64 %a18.cols, %a18.deps
  %a18.idx.i = mul i64 %a18.tmp4, %indvar.i
  %a18.tmp5 = add i64 %a18.tmp3, %a18.idx.i
  %a18.idx.j = mul i64 %a18.deps, %indvar.j
  %a18.tmp6 = add i64 %a18.tmp5, %a18.idx.j
  %a18.idx.k = add i64 %a18.tmp6, %indvar.k
  %a18.ptr = getelementptr float, float* %a18.data, i64 %a18.idx.k
  %a18.val = load float, float* %a18.ptr, align 4
  %sum.a18 = fadd float %sum.a17, %a18.val

  %a19.tmp1 = add i64 %a19.cols, 1
  %a19.tmp2 = mul i64 %a19.deps, %a19.tmp1
  %a19.tmp3 = add i64 %a19.tmp2, 1
  %a19.tmp4 = mul i64 %a19.cols, %a19.deps
  %a19.idx.i = mul i64 %a19.tmp4, %indvar.i
  %a19.tmp5 = add i64 %a19.tmp3, %a19.idx.i
  %a19.idx.j = mul i64 %a19.deps, %indvar.j
  %a19.tmp6  = add i64 %a19.tmp5, %a19.idx.j
  %a19.idx.k = add i64 %a19.tmp6, %indvar.k
  %a19.ptr = getelementptr float, float* %a19.data, i64 %a19.idx.k
  store float %sum.a18, float* %a19.ptr, align 4

  %indvar.k.next = add i64 %indvar.k, 1
  %a1.deps.sub = add i64 %a1.deps, -2
  %exitcond = icmp ne i64 %indvar.k.next, %a1.deps.sub
  br i1 %exitcond, label %for.body.k, label %for.inc.j

for.inc.j:
  %indvar.j.next = add i64 %indvar.j, 1
  %a1.cols.sub = add i64 %a1.cols, -2
  %exitcond.j = icmp ne i64 %indvar.j.next, %a1.cols.sub
  br i1 %exitcond.j, label %for.j, label %for.inc.i

for.inc.i:
  %indvar.i.next = add i64 %indvar.i, 1
  %a1.rows.sub = add i64 %a1.rows, -2
  %exitcond.i = icmp ne i64 %indvar.i.next, %a1.rows.sub
  br i1 %exitcond.i, label %for.i, label %for.inc.n

for.inc.n:
  %indvar.n.next = add nsw i64 %indvar.n, 1
  %exitcond.n = icmp ne i64 %indvar.n.next, %nn
  br i1 %exitcond.n, label %for.n, label %exit

exit:
  ret void
}
