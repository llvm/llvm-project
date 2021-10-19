; RUN: opt -S -loop-vectorize -instcombine -force-vector-width=2 -force-vector-interleave=1 -enable-interleaved-mem-accesses < %s | FileCheck %s
; RUN: opt -S -loop-vectorize -instcombine -force-vector-width=2 -force-vector-interleave=1 -enable-interleaved-mem-accesses -enable-masked-interleaved-mem-accesses < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
%pair = type { i64, i64 }

; Ensure that we vectorize the interleaved load group even though the loop
; contains a conditional store. The store group contains gaps and is not
; vectorized.
;
; CHECK-LABEL: @interleaved_with_cond_store_0(
;
; CHECK: vector.ph
; CHECK:   %n.mod.vf = and i64 %[[N:.+]], 1
; CHECK:   %[[IsZero:[a-zA-Z0-9]+]] = icmp eq i64 %n.mod.vf, 0
; CHECK:   %[[R:.+]] = select i1 %[[IsZero]], i64 2, i64 %n.mod.vf
; CHECK:   %n.vec = sub nsw i64 %[[N]], %[[R]]
;
; CHECK: vector.body:
; CHECK:   %wide.vec = load <4 x i64>, <4 x i64>* %{{.*}}
; CHECK:   %strided.vec = shufflevector <4 x i64> %wide.vec, <4 x i64> poison, <2 x i32> <i32 0, i32 2>
;
; CHECK: pred.store.if
; CHECK:   %[[X1:.+]] = extractelement <4 x i64> %wide.vec, i32 0
; CHECK:   store i64 %[[X1]], {{.*}}
;
; CHECK: pred.store.if
; CHECK:   %[[X2:.+]] = extractelement <4 x i64> %wide.vec, i32 2
; CHECK:   store i64 %[[X2]], {{.*}}

define void @interleaved_with_cond_store_0(%pair *%p, i64 %x, i64 %n) {
entry:
  br label %for.body

for.body:
  %i  = phi i64 [ %i.next, %if.merge ], [ 0, %entry ]
  %p.1 = getelementptr inbounds %pair, %pair* %p, i64 %i, i32 1
  %0 = load i64, i64* %p.1, align 8
  %1 = icmp eq i64 %0, %x
  br i1 %1, label %if.then, label %if.merge

if.then:
  store i64 %0, i64* %p.1, align 8
  br label %if.merge

if.merge:
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

; Ensure that we don't form a single interleaved group for the two loads. The
; conditional store prevents the second load from being hoisted. The two load
; groups are separately vectorized. The store group contains gaps and is not
; vectorized.
;
; CHECK-LABEL: @interleaved_with_cond_store_1(
;
; CHECK: vector.ph
; CHECK:   %n.mod.vf = and i64 %[[N:.+]], 1
; CHECK:   %[[IsZero:[a-zA-Z0-9]+]] = icmp eq i64 %n.mod.vf, 0
; CHECK:   %[[R:.+]] = select i1 %[[IsZero]], i64 2, i64 %n.mod.vf
; CHECK:   %n.vec = sub nsw i64 %[[N]], %[[R]]
;
; CHECK: vector.body:
; CHECK:   %[[L1:.+]] = load <4 x i64>, <4 x i64>* %{{.*}}
; CHECK:   %strided.vec = shufflevector <4 x i64> %[[L1]], <4 x i64> poison, <2 x i32> <i32 0, i32 2>
;
; CHECK: pred.store.if
; CHECK:   %[[X1:.+]] = extractelement <4 x i64> %wide.vec, i32 0
; CHECK:   store i64 %[[X1]], {{.*}}
;
; CHECK: pred.store.if
; CHECK:   %[[X2:.+]] = extractelement <4 x i64> %wide.vec, i32 2
; CHECK:   store i64 %[[X2]], {{.*}}
;
; CHECK: pred.store.continue
; CHECK:   %[[L2:.+]] = load <4 x i64>, <4 x i64>* {{.*}}
; CHECK:   %[[X3:.+]] = extractelement <4 x i64> %[[L2]], i32 0
; CHECK:   store i64 %[[X3]], {{.*}}
; CHECK:   %[[X4:.+]] = extractelement <4 x i64> %[[L2]], i32 2
; CHECK:   store i64 %[[X4]], {{.*}}

define void @interleaved_with_cond_store_1(%pair *%p, i64 %x, i64 %n) {
entry:
  br label %for.body

for.body:
  %i  = phi i64 [ %i.next, %if.merge ], [ 0, %entry ]
  %p.0 = getelementptr inbounds %pair, %pair* %p, i64 %i, i32 0
  %p.1 = getelementptr inbounds %pair, %pair* %p, i64 %i, i32 1
  %0 = load i64, i64* %p.1, align 8
  %1 = icmp eq i64 %0, %x
  br i1 %1, label %if.then, label %if.merge

if.then:
  store i64 %0, i64* %p.0, align 8
  br label %if.merge

if.merge:
  %2 = load i64, i64* %p.0, align 8
  store i64 %2, i64 *%p.1, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

; Ensure that we don't create a single interleaved group for the two stores.
; The second store is conditional and we can't sink the first store inside the
; predicated block. The load group is vectorized, and the store groups contain
; gaps and are not vectorized.
;
; CHECK-LABEL: @interleaved_with_cond_store_2(
;
; CHECK: vector.ph
; CHECK:   %n.mod.vf = and i64 %[[N:.+]], 1
; CHECK:   %[[IsZero:[a-zA-Z0-9]+]] = icmp eq i64 %n.mod.vf, 0
; CHECK:   %[[R:.+]] = select i1 %[[IsZero]], i64 2, i64 %n.mod.vf
; CHECK:   %n.vec = sub nsw i64 %[[N]], %[[R]]
;
; CHECK: vector.body:
; CHECK:   %[[L1:.+]] = load <4 x i64>, <4 x i64>* %{{.*}}
; CHECK:   %strided.vec = shufflevector <4 x i64> %[[L1]], <4 x i64> poison, <2 x i32> <i32 0, i32 2>
; CHECK:   store i64 %x, {{.*}}
; CHECK:   store i64 %x, {{.*}}
;
; CHECK: pred.store.if
; CHECK:   %[[X1:.+]] = extractelement <4 x i64> %wide.vec, i32 0
; CHECK:   store i64 %[[X1]], {{.*}}
;
; CHECK: pred.store.if
; CHECK:   %[[X2:.+]] = extractelement <4 x i64> %wide.vec, i32 2
; CHECK:   store i64 %[[X2]], {{.*}}

define void @interleaved_with_cond_store_2(%pair *%p, i64 %x, i64 %n) {
entry:
  br label %for.body

for.body:
  %i  = phi i64 [ %i.next, %if.merge ], [ 0, %entry ]
  %p.0 = getelementptr inbounds %pair, %pair* %p, i64 %i, i32 0
  %p.1 = getelementptr inbounds %pair, %pair* %p, i64 %i, i32 1
  %0 = load i64, i64* %p.1, align 8
  store i64 %x, i64* %p.0, align 8
  %1 = icmp eq i64 %0, %x
  br i1 %1, label %if.then, label %if.merge

if.then:
  store i64 %0, i64* %p.1, align 8
  br label %if.merge

if.merge:
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}
