; RUN: opt -S -passes='print<loopnest>' < %s 2>&1 > /dev/null | FileCheck %s

; int f(int N, int M) {
;   int res = 0;
;   for (int i = 0; i < N; ++i) {
;     for (int j = 0; j < M; ++j) res += i * j;
;   }
;   return res;
; }

define i32 @f(i32 %N, i32 %M) #0 {
; CHECK: IsPerfect=true, Depth=1, OutermostLoop: for.j, Loops: ( for.j )
; CHECK: IsPerfect=true, Depth=2, OutermostLoop: for.i, Loops: ( for.i for.j )
entry:
  %cmp4 = icmp slt i32 0, %N
  br i1 %cmp4, label %for.i.ph, label %for.i.end

for.i.ph:                                         ; preds = %entry
  br label %for.i

for.i:                                            ; preds = %for.i.ph, %for.i.inc
  %i.06 = phi i32 [ 0, %for.i.ph ], [ %inc5, %for.i.inc ]
  %res.05 = phi i32 [ 0, %for.i.ph ], [ %res.1.lcssa, %for.i.inc ]
  %cmp21 = icmp slt i32 0, %M
  br i1 %cmp21, label %for.j.ph, label %for.j.end

for.j.ph:                                         ; preds = %for.i
  br label %for.j

for.j:                                            ; preds = %for.j.ph, %for.j.inc
  %j.03 = phi i32 [ 0, %for.j.ph ], [ %inc, %for.j.inc ]
  %res.12 = phi i32 [ %res.05, %for.j.ph ], [ %add, %for.j.inc ]
  %mul = mul nsw i32 %i.06, %j.03
  %add = add nsw i32 %res.12, %mul
  br label %for.j.inc

for.j.inc:                                        ; preds = %for.j
  %inc = add nsw i32 %j.03, 1
  %cmp2 = icmp slt i32 %inc, %M
  br i1 %cmp2, label %for.j, label %for.j.end_crit_edge

for.j.end_crit_edge:                              ; preds = %for.j.inc
  %split = phi i32 [ %add, %for.j.inc ]
  br label %for.j.end

for.j.end:                                        ; preds = %for.j.end_crit_edge, %for.i
  %res.1.lcssa = phi i32 [ %split, %for.j.end_crit_edge ], [ %res.05, %for.i ]
  br label %for.i.inc

for.i.inc:                                        ; preds = %for.j.end
  %inc5 = add nsw i32 %i.06, 1
  %cmp = icmp slt i32 %inc5, %N
  br i1 %cmp, label %for.i, label %for.i.end_crit_edge

for.i.end_crit_edge:                              ; preds = %for.i.inc
  %split7 = phi i32 [ %res.1.lcssa, %for.i.inc ]
  br label %for.i.end

for.i.end:                                        ; preds = %for.i.end_crit_edge, %entry
  %res.0.lcssa = phi i32 [ %split7, %for.i.end_crit_edge ], [ 0, %entry ]
  ret i32 %res.0.lcssa
}

; int g(int N, int M, int K) {
;   int sum = 0, prod = 1;
;   for (int i = 0; i < N; ++i) {
;     for (int j = 0; j < M; ++j) {
;       for (int k = 0; k < K; ++k) {
;         sum += i * j * k;
;       }
;       prod *= (i + j);
;     }
;   }
;   return sum + prod;
; }
define i32 @g(i32 %N, i32 %M, i32 %K) #0 {
; CHECK: IsPerfect=true, Depth=1, OutermostLoop: for.k, Loops: ( for.k )
; CHECK: IsPerfect=false, Depth=2, OutermostLoop: for.j, Loops: ( for.j for.k )
; CHECK: IsPerfect=false, Depth=3, OutermostLoop: for.i, Loops: ( for.i for.j for.k )
entry:
  %cmp10 = icmp slt i32 0, %N
  br i1 %cmp10, label %for.i.ph, label %for.i.end

for.i.ph:                                         ; preds = %entry
  br label %for.i

for.i:                                            ; preds = %for.i.ph, %for.i.inc
  %i.013 = phi i32 [ 0, %for.i.ph ], [ %inc14, %for.i.inc ]
  %sum.012 = phi i32 [ 0, %for.i.ph ], [ %sum.1.lcssa, %for.i.inc ]
  %prod.011 = phi i32 [ 1, %for.i.ph ], [ %prod.1.lcssa, %for.i.inc ]
  %cmp24 = icmp slt i32 0, %M
  br i1 %cmp24, label %for.j.ph, label %for.j.end

for.j.ph:                                         ; preds = %for.i
  br label %for.j

for.j:                                            ; preds = %for.j.ph, %for.j.inc
  %j.07 = phi i32 [ 0, %for.j.ph ], [ %inc11, %for.j.inc ]
  %sum.16 = phi i32 [ %sum.012, %for.j.ph ], [ %sum.2.lcssa, %for.j.inc ]
  %prod.15 = phi i32 [ %prod.011, %for.j.ph ], [ %mul9, %for.j.inc ]
  %cmp51 = icmp slt i32 0, %K
  br i1 %cmp51, label %for.k.ph, label %for.k.end

for.k.ph:                                         ; preds = %for.j
  br label %for.k

for.k:                                            ; preds = %for.k.ph, %for.k.inc
  %k.03 = phi i32 [ 0, %for.k.ph ], [ %inc, %for.k.inc ]
  %sum.22 = phi i32 [ %sum.16, %for.k.ph ], [ %add, %for.k.inc ]
  %mul = mul nsw i32 %i.013, %j.07
  %mul7 = mul nsw i32 %mul, %k.03
  %add = add nsw i32 %sum.22, %mul7
  br label %for.k.inc

for.k.inc:                                        ; preds = %for.k
  %inc = add nsw i32 %k.03, 1
  %cmp5 = icmp slt i32 %inc, %K
  br i1 %cmp5, label %for.k, label %for.k.end_crit_edge

for.k.end_crit_edge:                              ; preds = %for.k.inc
  %split = phi i32 [ %add, %for.k.inc ]
  br label %for.k.end

for.k.end:                                        ; preds = %for.k.end_crit_edge, %for.j
  %sum.2.lcssa = phi i32 [ %split, %for.k.end_crit_edge ], [ %sum.16, %for.j ]
  %add8 = add nsw i32 %i.013, %j.07
  %mul9 = mul nsw i32 %prod.15, %add8
  br label %for.j.inc

for.j.inc:                                        ; preds = %for.k.end
  %inc11 = add nsw i32 %j.07, 1
  %cmp2 = icmp slt i32 %inc11, %M
  br i1 %cmp2, label %for.j, label %for.j.end_crit_edge

for.j.end_crit_edge:                              ; preds = %for.j.inc
  %split8 = phi i32 [ %mul9, %for.j.inc ]
  %split9 = phi i32 [ %sum.2.lcssa, %for.j.inc ]
  br label %for.j.end

for.j.end:                                        ; preds = %for.j.end1crit_edge, %for.i
  %prod.1.lcssa = phi i32 [ %split8, %for.j.end_crit_edge ], [ %prod.011, %for.i ]
  %sum.1.lcssa = phi i32 [ %split9, %for.j.end_crit_edge ], [ %sum.012, %for.i ]
  br label %for.i.inc

for.i.inc:                                        ; preds = %for.j.end
  %inc14 = add nsw i32 %i.013, 1
  %cmp = icmp slt i32 %inc14, %N
  br i1 %cmp, label %for.i, label %for.i.end_crit_edge

for.i.end_crit_edge:                              ; preds = %for.i.inc
  %split14 = phi i32 [ %prod.1.lcssa, %for.i.inc ]
  %split15 = phi i32 [ %sum.1.lcssa, %for.i.inc ]
  br label %for.i.end

for.i.end:                                        ; preds = %for.i.end_crit_edge, %entry
  %prod.0.lcssa = phi i32 [ %split14, %for.i.end_crit_edge ], [ 1, %entry ]
  %sum.0.lcssa = phi i32 [ %split15, %for.i.end_crit_edge ], [ 0, %entry ]
  %add16 = add nsw i32 %sum.0.lcssa, %prod.0.lcssa
  ret i32 %add16
}

; int h(int N, int M, int K) {
;   int sum = 0;
;   for (int i = 0; i < N; ++i) {
;     for (int j = 0; j < M; ++j) {
;       for (int k = 0; k < K; ++k) {
;         sum += i * j * k;
;       }
;     }
;   }
;   return sum;
; }
define i32 @h(i32 %N, i32 %M, i32 %K) #0 {
; CHECK: IsPerfect=true, Depth=1, OutermostLoop: for.k, Loops: ( for.k )
; CHECK: IsPerfect=true, Depth=2, OutermostLoop: for.j, Loops: ( for.j for.k )
; CHECK: IsPerfect=true, Depth=3, OutermostLoop: for.i, Loops: ( for.i for.j for.k )
entry:
  %cmp8 = icmp slt i32 0, %N
  br i1 %cmp8, label %for.i.ph, label %for.i.end

for.i.ph:                                         ; preds = %entry
  br label %for.i

for.i:                                            ; preds = %for.i.ph, %for.i.inc
  %i.010 = phi i32 [ 0, %for.i.ph ], [ %inc12, %for.i.inc ]
  %sum.09 = phi i32 [ 0, %for.i.ph ], [ %sum.1.lcssa, %for.i.inc ]
  %cmp24 = icmp slt i32 0, %M
  br i1 %cmp24, label %for.j.ph, label %for.j.end

for.j.ph:                                         ; preds = %for.i
  br label %for.j

for.j:                                            ; preds = %for.j.ph, %for.j.inc
  %j.06 = phi i32 [ 0, %for.j.ph ], [ %inc9, %for.j.inc ]
  %sum.15 = phi i32 [ %sum.09, %for.j.ph ], [ %sum.2.lcssa, %for.j.inc ]
  %cmp51 = icmp slt i32 0, %K
  br i1 %cmp51, label %for.k.ph, label %for.k.end

for.k.ph:                                         ; preds = %for.j
  br label %for.k

for.k:                                            ; preds = %for.k.ph, %for.k.inc
  %k.03 = phi i32 [ 0, %for.k.ph ], [ %inc, %for.k.inc ]
  %sum.22 = phi i32 [ %sum.15, %for.k.ph ], [ %add, %for.k.inc ]
  %mul = mul nsw i32 %i.010, %j.06
  %mul7 = mul nsw i32 %mul, %k.03
  %add = add nsw i32 %sum.22, %mul7
  br label %for.k.inc

for.k.inc:                                        ; preds = %for.k
  %inc = add nsw i32 %k.03, 1
  %cmp5 = icmp slt i32 %inc, %K
  br i1 %cmp5, label %for.k, label %for.k.end_crit_edge

for.k.end_crit_edge:                              ; preds = %for.k.inc
  %split = phi i32 [ %add, %for.k.inc ]
  br label %for.k.end

for.k.end:                                        ; preds = %for.k.end_crit_edge, %for.j
  %sum.2.lcssa = phi i32 [ %split, %for.k.end_crit_edge ], [ %sum.15, %for.j ]
  br label %for.j.inc

for.j.inc:                                        ; preds = %for.k.end
  %inc9 = add nsw i32 %j.06, 1
  %cmp2 = icmp slt i32 %inc9, %M
  br i1 %cmp2, label %for.j, label %for.j.end_crit_edge

for.j.end_crit_edge:                              ; preds = %for.j.inc
  %split7 = phi i32 [ %sum.2.lcssa, %for.j.inc ]
  br label %for.j.end

for.j.end:                                        ; preds = %for.j.end_crit_edge, %for.i
  %sum.1.lcssa = phi i32 [ %split7, %for.j.end_crit_edge ], [ %sum.09, %for.i ]
  br label %for.i.inc

for.i.inc:                                        ; preds = %for.j.end
  %inc12 = add nsw i32 %i.010, 1
  %cmp = icmp slt i32 %inc12, %N
  br i1 %cmp, label %for.i, label %for.i.end_crit_edge

for.i.end_crit_edge:                              ; preds = %for.i.inc
  %split11 = phi i32 [ %sum.1.lcssa, %for.i.inc ]
  br label %for.i.end

for.i.end:                                        ; preds = %for.i.end_crit_edge, %entry
  %sum.0.lcssa = phi i32 [ %split11, %for.i.end_crit_edge ], [ 0, %entry ]
  ret i32 %sum.0.lcssa
}
