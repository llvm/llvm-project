; RUN: opt -S -disable-output -passes='print<access-info>' < %s 2>&1 | FileCheck %s

; In the function below some of the accesses are done as float types and some
; are done as i32 types. When doing dependence analysis the type should not
; matter if it can be determined that they are the same size.

%int_pair = type { i32, i32 }

; CHECK-LABEL: function 'backdep_type_size_equivalence':
; CHECK-NEXT:    loop:
; CHECK-NEXT:      Memory dependences are safe with a maximum dependence distance of 800 bytes
; CHECK-NEXT:      Dependences:
; CHECK-NEXT:        Forward:
; CHECK-NEXT:            %ld.f32 = load float, ptr %gep.iv, align 8 ->
; CHECK-NEXT:            store i32 %indvars.iv.i32, ptr %gep.iv, align 8
; CHECK-EMPTY:
; CHECK-NEXT:        Forward:
; CHECK-NEXT:            %ld.f32 = load float, ptr %gep.iv, align 8 ->
; CHECK-NEXT:            store float %val, ptr %gep.iv.min.100, align 8
; CHECK-EMPTY:
; CHECK-NEXT:        BackwardVectorizable:
; CHECK-NEXT:            store float %val, ptr %gep.iv.min.100, align 8 ->
; CHECK-NEXT:            store i32 %indvars.iv.i32, ptr %gep.iv, align 8
; CHECK-EMPTY:
; CHECK-NEXT:        Run-time memory checks:
; CHECK-NEXT:        Grouped accesses:

define void @backdep_type_size_equivalence(ptr nocapture %vec, i64 %n) {
entry:
  br label %loop

loop:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %loop ]

  ;; Load from vec[indvars.iv].x as float
  %gep.iv = getelementptr inbounds %int_pair, ptr %vec, i64 %indvars.iv, i32 0
  %ld.f32 = load float, ptr %gep.iv, align 8
  %val = fmul fast float %ld.f32, 5.0

  ;; Store to vec[indvars.iv - 100].x as float
  %indvars.iv.min.100 = add nsw i64 %indvars.iv, -100
  %gep.iv.min.100 = getelementptr inbounds %int_pair, ptr %vec, i64 %indvars.iv.min.100, i32 0
  store float %val, ptr %gep.iv.min.100, align 8

  ;; Store to vec[indvars.iv].x as i32, creating a backward dependency between
  ;; the two stores with different element types but the same element size.
  %indvars.iv.i32 = trunc i64 %indvars.iv to i32
  store i32 %indvars.iv.i32, ptr %gep.iv, align 8

  ;; Store to vec[indvars.iv].y as i32, strided accesses should be independent
  ;; between the two stores with different element types but the same element size.
  %gep.iv.1 = getelementptr inbounds %int_pair, ptr %vec, i64 %indvars.iv, i32 1
  store i32 %indvars.iv.i32, ptr %gep.iv.1, align 8

  ;; Store to vec[indvars.iv + n].y as i32, to verify no dependence in the case
  ;; of unknown dependence distance.
  %indvars.iv.n = add nuw nsw i64 %indvars.iv, %n
  %gep.iv.n = getelementptr inbounds %int_pair, ptr %vec, i64 %indvars.iv.n, i32 1
  store i32 %indvars.iv.i32, ptr %gep.iv.n, align 8

  ;; Loop condition.
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cond = icmp eq i64 %indvars.iv.next, %n
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}

; In the function below one of the accesses is done as i19 type, which has a
; different store size than the i32 type, even though their alloc sizes are
; equivalent. This is a negative test to ensure that they are not analyzed as
; in the tests above.
;
; CHECK-LABEL: function 'backdep_type_store_size_equivalence':
; CHECK-NEXT:    loop:
; CHECK-NEXT:      Report: unsafe dependent memory operations in loop.
; CHECK-NEXT:      Unknown data dependence.
; CHECK-NEXT:      Dependences:
; CHECK-NEXT:        Unknown:
; CHECK-NEXT:            %ld.f32 = load float, ptr %gep.iv, align 8 ->
; CHECK-NEXT:            store i19 %indvars.iv.i19, ptr %gep.iv, align 8
; CHECK-EMPTY:
; CHECK-NEXT:        Run-time memory checks:
; CHECK-NEXT:        Grouped accesses:

define void @backdep_type_store_size_equivalence(ptr nocapture %vec, i64 %n) {
entry:
  br label %loop

loop:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %loop ]

  ;; Load from vec[indvars.iv].x as float
  %gep.iv = getelementptr inbounds %int_pair, ptr %vec, i64 %indvars.iv, i32 0
  %ld.f32 = load float, ptr %gep.iv, align 8
  %val = fmul fast float %ld.f32, 5.0

  ;; Store to vec[indvars.iv].x as i19.
  %indvars.iv.i19 = trunc i64 %indvars.iv to i19
  store i19 %indvars.iv.i19, ptr %gep.iv, align 8

  ;; Loop condition.
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cond = icmp eq i64 %indvars.iv.next, %n
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}

; In the function below some of the accesses are done as double types and some
; are done as i64 and i32 types. This is a negative test to ensure that they
; are not analyzed as in the tests above.

; CHECK-LABEL: function 'neg_dist_dep_type_size_equivalence':
; CHECK-NEXT:    loop:
; CHECK-NEXT:      Report: unsafe dependent memory operations in loop.
; CHECK-NEXT:      Unknown data dependence.
; CHECK-NEXT:      Dependences:
; CHECK-NEXT:        Unknown:
; CHECK-NEXT:            %ld.f64 = load double, ptr %gep.iv, align 8 ->
; CHECK-NEXT:		         store i32 %ld.i64.i32, ptr %gep.iv.n, align 8
; CHECK-EMPTY:
; CHECK-NEXT:        Unknown:
; CHECK-NEXT:            %ld.i64 = load i64, ptr %gep.iv, align 8 ->
; CHECK-NEXT:		         store i32 %ld.i64.i32, ptr %gep.iv.n, align 8
; CHECK-EMPTY:
; CHECK-NEXT:        BackwardVectorizableButPreventsForwarding:
; CHECK-NEXT:            %ld.f64 = load double, ptr %gep.iv, align 8 ->
; CHECK-NEXT:            store double %val, ptr %gep.iv.101, align 8
; CHECK-EMPTY:
; CHECK-NEXT:        ForwardButPreventsForwarding:
; CHECK-NEXT:            store double %val, ptr %gep.iv.101, align 8 ->
; CHECK-NEXT:            %ld.i64 = load i64, ptr %gep.iv, align 8
; CHECK-EMPTY:
; CHECK-NEXT:        Unknown:
; CHECK-NEXT:            store double %val, ptr %gep.iv.101, align 8 ->
; CHECK-NEXT:		         store i32 %ld.i64.i32, ptr %gep.iv.n, align 8
; CHECK-EMPTY:
; CHECK-NEXT:      Run-time memory checks:
; CHECK-NEXT:      Grouped accesses:

define void @neg_dist_dep_type_size_equivalence(ptr nocapture %vec, i64 %n) {
entry:
  br label %loop

loop:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %loop ]

  ;; Load from vec[indvars.iv] as double
  %gep.iv = getelementptr i64, ptr %vec, i64 %indvars.iv
  %ld.f64 = load double, ptr %gep.iv, align 8
  %val = fmul fast double %ld.f64, 5.0

  ;; Store to vec[indvars.iv + 101] as double
  %indvars.iv.101 = add nsw i64 %indvars.iv, 101
  %gep.iv.101 = getelementptr i64, ptr %vec, i64 %indvars.iv.101
  store double %val, ptr %gep.iv.101, align 8

  ;; Read from vec[indvars.iv] as i64 creating
  ;; a forward but prevents forwarding dependence
  ;; with different types but same sizes.
  %ld.i64 = load i64, ptr %gep.iv, align 8

  ;; Different sizes
  %indvars.iv.n = add nuw nsw i64 %indvars.iv, %n
  %gep.iv.n = getelementptr inbounds i64, ptr %vec, i64 %indvars.iv.n
  %ld.i64.i32 = trunc i64 %ld.i64 to i32
  store i32 %ld.i64.i32, ptr %gep.iv.n, align 8

  ;; Loop condition.
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cond = icmp eq i64 %indvars.iv.next, %n
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}
