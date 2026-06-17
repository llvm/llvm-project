; RUN: opt < %s -cache-line-size=64 -passes='print<loop-cache-cost>' -disable-output 2>&1 | FileCheck %s

;; A partially perfect nest is decomposed into disjoint linear chains, each
;; costed against its own innermost loop. A loop that forks the nest, or that
;; encloses a fork, is not a reorder candidate and is omitted.

;; for (i) {
;;   for (j)
;;     for (jj) A[jj][j] = 1;
;;   for (k)
;;     for (kk) B[k][kk] = 1;
;; }
;; 'i' forks, so it is omitted. Chains {j,jj} (A[jj][j]) and {k,kk} (B[k][kk])
;; are costed independently.

; CHECK-DAG: Loop 'a.jj' has cost = 1000000
; CHECK-DAG: Loop 'a.j' has cost = 130000
; CHECK-DAG: Loop 'a.k' has cost = 1000000
; CHECK-DAG: Loop 'a.kk' has cost = 130000
; CHECK-NOT: Loop 'a.i'

define void @fork_at_top(ptr %A, ptr %B) {
entry:
  br label %a.i
a.i:
  %iv = phi i64 [ 0, %entry ], [ %iv.n, %a.i.inc ]
  br label %a.j
a.j:
  %jv = phi i64 [ 0, %a.i ], [ %jv.n, %a.j.inc ]
  br label %a.jj
a.jj:
  %jjv = phi i64 [ 0, %a.j ], [ %jjv.n, %a.jj ]
  %ga = getelementptr inbounds [100 x [100 x double]], ptr %A, i64 0, i64 %jjv, i64 %jv
  store double 1.0, ptr %ga
  %jjv.n = add nsw i64 %jjv, 1
  %jj.e = icmp eq i64 %jjv.n, 100
  br i1 %jj.e, label %a.j.inc, label %a.jj
a.j.inc:
  %jv.n = add nsw i64 %jv, 1
  %j.e = icmp eq i64 %jv.n, 100
  br i1 %j.e, label %a.k, label %a.j
a.k:
  %kv = phi i64 [ 0, %a.j.inc ], [ %kv.n, %a.k.inc ]
  br label %a.kk
a.kk:
  %kkv = phi i64 [ 0, %a.k ], [ %kkv.n, %a.kk ]
  %gb = getelementptr inbounds [100 x [100 x double]], ptr %B, i64 0, i64 %kv, i64 %kkv
  store double 1.0, ptr %gb
  %kkv.n = add nsw i64 %kkv, 1
  %kk.e = icmp eq i64 %kkv.n, 100
  br i1 %kk.e, label %a.k.inc, label %a.kk
a.k.inc:
  %kv.n = add nsw i64 %kv, 1
  %k.e = icmp eq i64 %kv.n, 100
  br i1 %k.e, label %a.i.inc, label %a.k
a.i.inc:
  %iv.n = add nsw i64 %iv, 1
  %i.e = icmp eq i64 %iv.n, 100
  br i1 %i.e, label %exit, label %a.i
exit:
  ret void
}

;; for (i)
;;   for (j) {
;;     for (k) C[k] = 1;
;;     for (l) D[l] = 1;
;;   }
;; 'j' forks, so both 'i' and 'j' are omitted. The singleton chains {k} (C[k])
;; and {l} (D[l]) are costed independently.

; CHECK-DAG: Loop 'b.k' has cost = 130000
; CHECK-DAG: Loop 'b.l' has cost = 130000
; CHECK-NOT: Loop 'b.i'
; CHECK-NOT: Loop 'b.j'

define void @fork_in_middle(ptr %C, ptr %D) {
entry:
  br label %b.i
b.i:
  %iv = phi i64 [ 0, %entry ], [ %iv.n, %b.i.inc ]
  br label %b.j
b.j:
  %jv = phi i64 [ 0, %b.i ], [ %jv.n, %b.j.inc ]
  br label %b.k
b.k:
  %kv = phi i64 [ 0, %b.j ], [ %kv.n, %b.k ]
  %gc = getelementptr inbounds double, ptr %C, i64 %kv
  store double 1.0, ptr %gc
  %kv.n = add nsw i64 %kv, 1
  %k.e = icmp eq i64 %kv.n, 100
  br i1 %k.e, label %b.l, label %b.k
b.l:
  %lv = phi i64 [ 0, %b.k ], [ %lv.n, %b.l ]
  %gd = getelementptr inbounds double, ptr %D, i64 %lv
  store double 1.0, ptr %gd
  %lv.n = add nsw i64 %lv, 1
  %l.e = icmp eq i64 %lv.n, 100
  br i1 %l.e, label %b.j.inc, label %b.l
b.j.inc:
  %jv.n = add nsw i64 %jv, 1
  %j.e = icmp eq i64 %jv.n, 100
  br i1 %j.e, label %b.i.inc, label %b.j
b.i.inc:
  %iv.n = add nsw i64 %iv, 1
  %i.e = icmp eq i64 %iv.n, 100
  br i1 %i.e, label %exit, label %b.i
exit:
  ret void
}

;; for (i) {
;;   for (j)  for (jj) A[jj][j] = 1;
;;   for (k)  for (kk) B[k][kk] = 1;
;;   for (l)  for (ll) C[ll][l] = 1;
;; }
;; Three-way fork at the outermost loop: confirms more than two chains are
;; produced, costed independently. 'c.i' is omitted.

; CHECK-DAG: Loop 'c.jj' has cost = 1000000
; CHECK-DAG: Loop 'c.j' has cost = 130000
; CHECK-DAG: Loop 'c.k' has cost = 1000000
; CHECK-DAG: Loop 'c.kk' has cost = 130000
; CHECK-DAG: Loop 'c.ll' has cost = 1000000
; CHECK-DAG: Loop 'c.l' has cost = 130000
; CHECK-NOT: Loop 'c.i'

define void @fork_three_way(ptr %A, ptr %B, ptr %C) {
entry:
  br label %c.i
c.i:
  %iv = phi i64 [ 0, %entry ], [ %iv.n, %c.i.inc ]
  br label %c.j
c.j:
  %jv = phi i64 [ 0, %c.i ], [ %jv.n, %c.j.inc ]
  br label %c.jj
c.jj:
  %jjv = phi i64 [ 0, %c.j ], [ %jjv.n, %c.jj ]
  %ga = getelementptr inbounds [100 x [100 x double]], ptr %A, i64 0, i64 %jjv, i64 %jv
  store double 1.0, ptr %ga
  %jjv.n = add nsw i64 %jjv, 1
  %jj.e = icmp eq i64 %jjv.n, 100
  br i1 %jj.e, label %c.j.inc, label %c.jj
c.j.inc:
  %jv.n = add nsw i64 %jv, 1
  %j.e = icmp eq i64 %jv.n, 100
  br i1 %j.e, label %c.k, label %c.j
c.k:
  %kv = phi i64 [ 0, %c.j.inc ], [ %kv.n, %c.k.inc ]
  br label %c.kk
c.kk:
  %kkv = phi i64 [ 0, %c.k ], [ %kkv.n, %c.kk ]
  %gb = getelementptr inbounds [100 x [100 x double]], ptr %B, i64 0, i64 %kv, i64 %kkv
  store double 1.0, ptr %gb
  %kkv.n = add nsw i64 %kkv, 1
  %kk.e = icmp eq i64 %kkv.n, 100
  br i1 %kk.e, label %c.k.inc, label %c.kk
c.k.inc:
  %kv.n = add nsw i64 %kv, 1
  %k.e = icmp eq i64 %kv.n, 100
  br i1 %k.e, label %c.l, label %c.k
c.l:
  %lv = phi i64 [ 0, %c.k.inc ], [ %lv.n, %c.l.inc ]
  br label %c.ll
c.ll:
  %llv = phi i64 [ 0, %c.l ], [ %llv.n, %c.ll ]
  %gc = getelementptr inbounds [100 x [100 x double]], ptr %C, i64 0, i64 %llv, i64 %lv
  store double 1.0, ptr %gc
  %llv.n = add nsw i64 %llv, 1
  %ll.e = icmp eq i64 %llv.n, 100
  br i1 %ll.e, label %c.l.inc, label %c.ll
c.l.inc:
  %lv.n = add nsw i64 %lv, 1
  %l.e = icmp eq i64 %lv.n, 100
  br i1 %l.e, label %c.i.inc, label %c.l
c.i.inc:
  %iv.n = add nsw i64 %iv, 1
  %i.e = icmp eq i64 %iv.n, 100
  br i1 %i.e, label %exit, label %c.i
exit:
  ret void
}

;; for (i) {
;;   for (j)  for (jj) A[jj][j] = 1;
;;   for (k)            D[k] = 1;
;; }
;; Asymmetric depths under a fork: chains of different sizes ({j,jj} and {k})
;; produced from the same fork. 'd.i' is omitted.

; CHECK-DAG: Loop 'd.jj' has cost = 1000000
; CHECK-DAG: Loop 'd.j' has cost = 130000
; CHECK-DAG: Loop 'd.k' has cost = 1300
; CHECK-NOT: Loop 'd.i'

define void @fork_asymmetric_depths(ptr %A, ptr %D) {
entry:
  br label %d.i
d.i:
  %iv = phi i64 [ 0, %entry ], [ %iv.n, %d.i.inc ]
  br label %d.j
d.j:
  %jv = phi i64 [ 0, %d.i ], [ %jv.n, %d.j.inc ]
  br label %d.jj
d.jj:
  %jjv = phi i64 [ 0, %d.j ], [ %jjv.n, %d.jj ]
  %ga = getelementptr inbounds [100 x [100 x double]], ptr %A, i64 0, i64 %jjv, i64 %jv
  store double 1.0, ptr %ga
  %jjv.n = add nsw i64 %jjv, 1
  %jj.e = icmp eq i64 %jjv.n, 100
  br i1 %jj.e, label %d.j.inc, label %d.jj
d.j.inc:
  %jv.n = add nsw i64 %jv, 1
  %j.e = icmp eq i64 %jv.n, 100
  br i1 %j.e, label %d.k, label %d.j
d.k:
  %kv = phi i64 [ 0, %d.j.inc ], [ %kv.n, %d.k ]
  %gd = getelementptr inbounds double, ptr %D, i64 %kv
  store double 1.0, ptr %gd
  %kv.n = add nsw i64 %kv, 1
  %k.e = icmp eq i64 %kv.n, 100
  br i1 %k.e, label %d.i.inc, label %d.k
d.i.inc:
  %iv.n = add nsw i64 %iv, 1
  %i.e = icmp eq i64 %iv.n, 100
  br i1 %i.e, label %exit, label %d.i
exit:
  ret void
}
