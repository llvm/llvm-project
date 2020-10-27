; RUN: opt < %s -analyze -scalar-evolution | FileCheck %s

; CHECK-LABEL @test1
; CHECK       %add.lcssa.wide = phi i64 [ %indvars.iv.next, %do.body ]
; CHECK-NEXT  -->  %add.lcssa.wide U: [1,2147483648) S: [1,2147483648)

define i64 @test1(i32 signext %n, float* %A) {
entry:
  %0 = sext i32 %n to i64
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %do.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %A, i64 %indvars.iv
  store float 1.000000e+00, float* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp = icmp slt i64 %indvars.iv.next, %0
  br i1 %cmp, label %do.body, label %do.end

do.end:                                           ; preds = %do.body
  %add.lcssa.wide = phi i64 [ %indvars.iv.next, %do.body ]
  ret i64 %add.lcssa.wide
}

; CHECK-LABEL @test2
; CHECK:      %tmp24 = phi i64 [ %tmp14, %bb22 ], [ %tmp14, %bb13 ]
; CHECK-NEXT: -->  %tmp24 U: full-set S: full-set       Exits: <<Unknown>>      LoopDispositions: { %bb13: Variant, %bb8: Variant, %bb17: Invariant, %bb27: Invariant }

define void @test2(i64 %arg, i32* noalias %arg1) {
bb:
  %tmp = icmp slt i64 0, %arg
  br i1 %tmp, label %bb7, label %bb48

bb7:                                              ; preds = %bb
  br label %bb8

bb8:                                              ; preds = %bb44, %bb7
  %tmp9 = phi i64 [ 0, %bb7 ], [ %tmp45, %bb44 ]
  %tmp10 = add nsw i64 %arg, -1
  %tmp11 = icmp slt i64 1, %tmp10
  br i1 %tmp11, label %bb12, label %bb43

bb12:                                             ; preds = %bb8
  br label %bb13

bb13:                                             ; preds = %bb39, %bb12
  %tmp14 = phi i64 [ 1, %bb12 ], [ %tmp40, %bb39 ]
  %tmp15 = icmp slt i64 0, %arg
  br i1 %tmp15, label %bb16, label %bb23

bb16:                                             ; preds = %bb13
  br label %bb17

bb17:                                             ; preds = %bb19, %bb16
  %tmp18 = phi i64 [ 0, %bb16 ], [ %tmp20, %bb19 ]
  br label %bb19

bb19:                                             ; preds = %bb17
  %tmp20 = add nuw nsw i64 %tmp18, 1
  %tmp21 = icmp slt i64 %tmp20, %arg
  br i1 %tmp21, label %bb17, label %bb22

bb22:                                             ; preds = %bb19
  br label %bb23

bb23:                                             ; preds = %bb22, %bb13
  %tmp24 = phi i64 [ %tmp14, %bb22 ], [ %tmp14, %bb13 ]
  %tmp25 = icmp slt i64 0, %arg
  br i1 %tmp25, label %bb26, label %bb37

bb26:                                             ; preds = %bb23
  br label %bb27

bb27:                                             ; preds = %bb33, %bb26
  %tmp28 = phi i64 [ 0, %bb26 ], [ %tmp34, %bb33 ]
  %tmp29 = mul nsw i64 %tmp9, %arg
  %tmp30 = getelementptr inbounds i32, i32* %arg1, i64 %tmp24
  %tmp31 = getelementptr inbounds i32, i32* %tmp30, i64 %tmp29
  %tmp32 = load i32, i32* %tmp31, align 4
  br label %bb33

bb33:                                             ; preds = %bb27
  %tmp34 = add nuw nsw i64 %tmp28, 1
  %tmp35 = icmp slt i64 %tmp34, %arg
  br i1 %tmp35, label %bb27, label %bb36

bb36:                                             ; preds = %bb33
  br label %bb37

bb37:                                             ; preds = %bb36, %bb23
  %tmp38 = phi i64 [ %tmp24, %bb36 ], [ %tmp24, %bb23 ]
  br label %bb39

bb39:                                             ; preds = %bb37
  %tmp40 = add nuw nsw i64 %tmp38, 1
  %tmp41 = icmp slt i64 %tmp40, %tmp10
  br i1 %tmp41, label %bb13, label %bb42

bb42:                                             ; preds = %bb39
  br label %bb43

bb43:                                             ; preds = %bb42, %bb8
  br label %bb44

bb44:                                             ; preds = %bb43
  %tmp45 = add nuw nsw i64 %tmp9, 1
  %tmp46 = icmp slt i64 %tmp45, %arg
  br i1 %tmp46, label %bb8, label %bb47

bb47:                                             ; preds = %bb44
  br label %bb48

bb48:                                             ; preds = %bb47, %bb
  ret void
}

; CHECK-LABEL @test3

; CHECK:      %tmp14 = phi i64 [ %tmp40, %bb39 ], [ 1, %bb8 ]
; CHECK-NEXT: -->  {1,+,1}<%bb13> U: [1,9223372036854775807) S: [1,9223372036854775807)
; CHECK-SAME:      Exits: (-2 + %arg)       LoopDispositions: { %bb13: Computable, %bb8: Variant, %bb17_a: Invariant, %bb27: Invariant }
; CHECK:      %tmp18 = phi i64 [ %tmp20, %bb17 ], [ 0, %bb13 ]
; CHECK-NEXT: -->  {0,+,1}<nuw><nsw><%bb17_a> U: [0,9223372036854775807) S: [0,9223372036854775807)
; CHECK-SAME:      Exits: (-1 + %arg)       LoopDispositions: { %bb17_a: Computable, %bb13: Variant, %bb8: Variant }

; CHECK:      %tmp24 = phi i64 [ %tmp14, %bb13 ], [ %tmp14, %bb17 ]
; CHECK-NEXT: -->  {1,+,1}<%bb13> U: [1,9223372036854775807) S: [1,9223372036854775807)
; CHECK-SAME:      Exits: (-2 + %arg)       LoopDispositions: { %bb13: Computable, %bb8: Variant, %bb17_a: Invariant, %bb27: Invariant }
; CHECK:       %tmp28 = phi i64 [ %tmp34, %bb27 ], [ 0, %bb23 ]
; CHECK-NEXT:  -->  {0,+,1}<nuw><nsw><%bb27> U: [0,9223372036854775807) S: [0,9223372036854775807)
; CHECK-SAME:       Exits: (-1 + %arg)      LoopDispositions: { %bb27: Computable, %bb13: Variant, %bb8: Variant }

; CHECK:      %tmp38 = phi i64 [ %tmp24, %bb23 ], [ %tmp24, %bb27 ]
; CHECK-NEXT: -->  {1,+,1}<%bb13> U: [1,9223372036854775807) S: [1,9223372036854775807)
; CHECK-SAME:      Exits: (-2 + %arg)       LoopDispositions: { %bb13: Computable, %bb8: Variant, %bb17_a: Invariant, %bb27: Invariant }

define void @test3(i64 %arg, i32* %arg1) {
bb:
  %tmp = icmp slt i64 0, %arg
  br i1 %tmp, label %bb8, label %bb48

bb8:                                              ; preds = %bb, %bb44
  %tmp9 = phi i64 [ %tmp45, %bb44 ], [ 0, %bb ]
  %tmp10 = add nsw i64 %arg, -1
  %tmp11 = icmp slt i64 1, %tmp10
  br i1 %tmp11, label %bb13, label %bb44

bb13:                                             ; preds = %bb8, %bb39
  %tmp14 = phi i64 [ %tmp40, %bb39 ], [ 1, %bb8 ]
  %tmp15 = icmp slt i64 0, %arg
  br i1 %tmp15, label %bb17_a, label %bb23

bb17_a:
  %tmp18 = phi i64 [ %tmp20, %bb17 ], [ 0, %bb13 ]
  %tmp20 = add nuw nsw i64 %tmp18, 1

  br label %bb17

bb17:                                             ; preds = %bb13, %bb17
  %tmp21 = icmp slt i64 %tmp20, %arg
  br i1 %tmp21, label %bb17_a, label %bb23

bb23:                                             ; preds = %bb17, %bb13
  %tmp24 = phi i64 [ %tmp14, %bb13 ], [ %tmp14, %bb17 ]
  %tmp25 = icmp slt i64 0, %arg
  br i1 %tmp25, label %bb27, label %bb39

bb27:                                             ; preds = %bb23, %bb27
  %tmp28 = phi i64 [ %tmp34, %bb27 ], [ 0, %bb23 ]
  %tmp29 = mul nsw i64 %tmp9, %arg
  %tmp30 = getelementptr inbounds i32, i32* %arg1, i64 %tmp24
  %tmp31 = getelementptr inbounds i32, i32* %tmp30, i64 %tmp29
  %tmp32 = load i32, i32* %tmp31, align 4
  %tmp34 = add nuw nsw i64 %tmp28, 1
  %tmp35 = icmp slt i64 %tmp34, %arg
  br i1 %tmp35, label %bb27, label %bb39

bb39:                                             ; preds = %bb23, %bb27
  %tmp38 = phi i64 [ %tmp24, %bb23 ], [ %tmp24, %bb27 ]
  %tmp40 = add nuw nsw i64 %tmp38, 1
  %tmp41 = icmp slt i64 %tmp40, %tmp10
  br i1 %tmp41, label %bb13, label %bb44

bb44:                                             ; preds = %bb8, %bb39
  %tmp45 = add nuw nsw i64 %tmp9, 1
  %tmp46 = icmp slt i64 %tmp45, %arg
  br i1 %tmp46, label %bb8, label %bb48

bb48:                                             ; preds = %bb44, %bb
  ret void
}
