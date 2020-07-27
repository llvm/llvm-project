; RUN: opt -S -loop-fusion -loop-fusion-peel-max-count=3 < %s | FileCheck %s

; Tests whether we can fuse two loops together if they have constant but a
; different tripcount.
; The first three iterations of the first loop should be peeled, and then the
; two loops should be fused together in this example.

; C Code
;
; int B[1024];
;
; void function(int *arg) {
;   for (int i = 0; i != 100; ++i)
;     arg[i] = ((i - 3)*(i+3)) % i;
;
;   for (int i = 3; i != 100; ++i)
;     B[i] = ((i-6)*(i+3)) % i;
; }

; CHECK-LABEL: void @function(i32* noalias %arg)
; CHECK-NEXT:  for.first.preheader:
; CHECK-NEXT:    br label %for.first.peel.begin
; CHECK:       for.first.peel.begin:
; CHECK-NEXT:    br label %for.first.peel
; CHECK:       for.first.peel:
; CHECK:         br label %for.first.latch.peel
; CHECK:       for.first.latch.peel:
; CHECK:         br label %for.first.peel.next
; CHECK:       for.first.peel.next:
; CHECK-NEXT:    br label %for.first.peel2
; CHECK:       for.first.peel2:
; CHECK:         br label %for.first.latch.peel10
; CHECK:       for.first.latch.peel10:
; CHECK:         br label %for.first.peel.next1
; CHECK:       for.first.peel.next1:
; CHECK-NEXT:    br label %for.first.peel15
; CHECK:       for.first.peel15:
; CHECK:         br label %for.first.latch.peel23
; CHECK:       for.first.latch.peel23:
; CHECK:         br label %for.first.peel.next14
; CHECK:       for.first.peel.next14:
; CHECK-NEXT:    br label %for.first.peel.next27
; CHECK:       for.first.peel.next27:
; CHECK-NEXT:    br label %for.first.preheader.peel.newph
; CHECK:       for.first.preheader.peel.newph:
; CHECK-NEXT:    br label %for.first
; CHECK:       for.first:
; CHECK:         br label %for.first.latch
; CHECK:       for.first.latch:
; CHECK:         br label %for.second.latch
; CHECK:       for.second.latch:
; CHECK:         br i1 %exitcond, label %for.first, label %for.end
; CHECK:       for.end:
; CHECK-NEXT:    ret void

@B = common global [1024 x i32] zeroinitializer, align 16

define void @function(i32* noalias %arg) {
for.first.preheader:
  br label %for.first

for.first:                                       ; preds = %for.first.preheader, %for.first.latch
  %.014 = phi i32 [ 0, %for.first.preheader ], [ %tmp15, %for.first.latch ]
  %indvars.iv23 = phi i64 [ 0, %for.first.preheader ], [ %indvars.iv.next3, %for.first.latch ]
  %tmp = add nsw i32 %.014, -3
  %tmp8 = add nuw nsw i64 %indvars.iv23, 3
  %tmp9 = trunc i64 %tmp8 to i32
  %tmp10 = mul nsw i32 %tmp, %tmp9
  %tmp11 = trunc i64 %indvars.iv23 to i32
  %tmp12 = srem i32 %tmp10, %tmp11
  %tmp13 = getelementptr inbounds i32, i32* %arg, i64 %indvars.iv23
  store i32 %tmp12, i32* %tmp13, align 4
  br label %for.first.latch

for.first.latch:                                 ; preds = %for.first
  %indvars.iv.next3 = add nuw nsw i64 %indvars.iv23, 1
  %tmp15 = add nuw nsw i32 %.014, 1
  %exitcond4 = icmp ne i64 %indvars.iv.next3, 100
  br i1 %exitcond4, label %for.first, label %for.second.preheader

for.second.preheader:                            ; preds = %for.first.latch
  br label %for.second

for.second:                                      ; preds = %for.second.preheader, %for.second.latch
  %.02 = phi i32 [ 0, %for.second.preheader ], [ %tmp28, %for.second.latch ]
  %indvars.iv1 = phi i64 [ 3, %for.second.preheader ], [ %indvars.iv.next, %for.second.latch ]
  %tmp20 = add nsw i32 %.02, -3
  %tmp21 = add nuw nsw i64 %indvars.iv1, 3
  %tmp22 = trunc i64 %tmp21 to i32
  %tmp23 = mul nsw i32 %tmp20, %tmp22
  %tmp24 = trunc i64 %indvars.iv1 to i32
  %tmp25 = srem i32 %tmp23, %tmp24
  %tmp26 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %indvars.iv1
  store i32 %tmp25, i32* %tmp26, align 4
  br label %for.second.latch

for.second.latch:                                ; preds = %for.second
  %indvars.iv.next = add nuw nsw i64 %indvars.iv1, 1
  %tmp28 = add nuw nsw i32 %.02, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 100
  br i1 %exitcond, label %for.second, label %for.end

for.end:                                        ; preds = %for.second.latch
  ret void
}
