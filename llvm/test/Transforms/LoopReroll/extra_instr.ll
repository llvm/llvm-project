; RUN: opt -opaque-pointers=0 -S -passes=loop-reroll   %s | FileCheck %s
target triple = "aarch64--linux-gnu"

define void @rerollable1([2 x i32]* nocapture %a) {
entry:
  br label %loop

loop:

; CHECK-LABEL: loop:
; CHECK-NEXT:    %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
; CHECK-NEXT:    [[SCEVGEP1:%.*]] = getelementptr [2 x i32], [2 x i32]* %a, i64 20, i64 %iv
; CHECK-NEXT:    [[SCEVGEP2:%.*]] = getelementptr [2 x i32], [2 x i32]* %a, i64 10, i64 %iv
; CHECK-NEXT:    [[VALUE:%.*]] = load i32, i32* [[SCEVGEP1]], align 4
; CHECK-NEXT:    store i32 [[VALUE]], i32* [[SCEVGEP2]], align 4

  ; base instruction
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]

  ; NO unrerollable instructions

  ; extra simple arithmetic operations, used by root instructions
  %plus20 = add nuw nsw i64 %iv, 20
  %plus10 = add nuw nsw i64 %iv, 10

  ; root instruction 0
  %ldptr0 = getelementptr inbounds [2 x i32], [2 x i32]* %a, i64 %plus20, i64 0
  %value0 = load i32, i32* %ldptr0, align 4
  %stptr0 = getelementptr inbounds [2 x i32], [2 x i32]* %a, i64 %plus10, i64 0
  store i32 %value0, i32* %stptr0, align 4

  ; root instruction 1
  %ldptr1 = getelementptr inbounds [2 x i32], [2 x i32]* %a, i64 %plus20, i64 1
  %value1 = load i32, i32* %ldptr1, align 4
  %stptr1 = getelementptr inbounds [2 x i32], [2 x i32]* %a, i64 %plus10, i64 1
  store i32 %value1, i32* %stptr1, align 4

  ; loop-increment
  %iv.next = add nuw nsw i64 %iv, 1

  ; latch
  %exitcond = icmp eq i64 %iv.next, 5
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @unrerollable1([2 x i32]* nocapture %a) {
entry:
  br label %loop

loop:

; CHECK-LABEL: loop:
; CHECK-NEXT:    %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
; CHECK-NEXT:    %stptrx = getelementptr inbounds [2 x i32], [2 x i32]* %a, i64 %iv, i64 0
; CHECK-NEXT:    store i32 999, i32* %stptrx, align 4

  ; base instruction
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]

  ; unrerollable instructions using %iv
  %stptrx = getelementptr inbounds [2 x i32], [2 x i32]* %a, i64 %iv, i64 0
  store i32 999, i32* %stptrx, align 4

  ; extra simple arithmetic operations, used by root instructions
  %plus20 = add nuw nsw i64 %iv, 20
  %plus10 = add nuw nsw i64 %iv, 10

  ; root instruction 0
  %ldptr0 = getelementptr inbounds [2 x i32], [2 x i32]* %a, i64 %plus20, i64 0
  %value0 = load i32, i32* %ldptr0, align 4
  %stptr0 = getelementptr inbounds [2 x i32], [2 x i32]* %a, i64 %plus10, i64 0
  store i32 %value0, i32* %stptr0, align 4

  ; root instruction 1
  %ldptr1 = getelementptr inbounds [2 x i32], [2 x i32]* %a, i64 %plus20, i64 1
  %value1 = load i32, i32* %ldptr1, align 4
  %stptr1 = getelementptr inbounds [2 x i32], [2 x i32]* %a, i64 %plus10, i64 1
  store i32 %value1, i32* %stptr1, align 4

  ; loop-increment
  %iv.next = add nuw nsw i64 %iv, 1

  ; latch
  %exitcond = icmp eq i64 %iv.next, 5
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @unrerollable2([2 x i32]* nocapture %a) {
entry:
  br label %loop

loop:

; CHECK-LABEL: loop:
; CHECK-NEXT:    %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
; CHECK-NEXT:    %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:    %stptrx = getelementptr inbounds [2 x i32], [2 x i32]* %a, i64 %iv.next, i64 0
; CHECK-NEXT:    store i32 999, i32* %stptrx, align 4

  ; base instruction
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]

  ; loop-increment
  %iv.next = add nuw nsw i64 %iv, 1

  ; unrerollable instructions using %iv.next
  %stptrx = getelementptr inbounds [2 x i32], [2 x i32]* %a, i64 %iv.next, i64 0
  store i32 999, i32* %stptrx, align 4

  ; extra simple arithmetic operations, used by root instructions
  %plus20 = add nuw nsw i64 %iv, 20
  %plus10 = add nuw nsw i64 %iv, 10

  ; root instruction 0
  %ldptr0 = getelementptr inbounds [2 x i32], [2 x i32]* %a, i64 %plus20, i64 0
  %value0 = load i32, i32* %ldptr0, align 4
  %stptr0 = getelementptr inbounds [2 x i32], [2 x i32]* %a, i64 %plus10, i64 0
  store i32 %value0, i32* %stptr0, align 4

  ; root instruction 1
  %ldptr1 = getelementptr inbounds [2 x i32], [2 x i32]* %a, i64 %plus20, i64 1
  %value1 = load i32, i32* %ldptr1, align 4
  %stptr1 = getelementptr inbounds [2 x i32], [2 x i32]* %a, i64 %plus10, i64 1
  store i32 %value1, i32* %stptr1, align 4

  ; latch
  %exitcond = icmp eq i64 %iv.next, 5
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define dso_local void @rerollable2() {
entry:
  br label %loop

loop:

; CHECK-LABEL: loop:
; CHECK-NEXT:    %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
; CHECK-NEXT:    {{%.*}} = add i32 %iv, {{20|24}}
; CHECK-NEXT:    {{%.*}} = add i32 %iv, {{20|24}}

  ; induction variable
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]

  ; scale instruction
  %iv.mul3 = mul nuw nsw i32 %iv, 3

  ; extra simple arithmetic operations, used by root instructions
  %iv.scaled = add nuw nsw i32 %iv.mul3, 20

  ; NO unrerollable instructions

  ; root set 1

  ; base instruction
  %iv.scaled.div5 = udiv i32 %iv.scaled, 5
  tail call void @bar(i32 %iv.scaled.div5)
  ; root instruction 0
  %iv.scaled.add1 = add nuw nsw i32 %iv.scaled, 1
  %iv.scaled.add1.div5 = udiv i32 %iv.scaled.add1, 5
  tail call void @bar(i32 %iv.scaled.add1.div5)
  ; root instruction 2
  %iv.scaled.add2 = add nuw nsw i32 %iv.scaled, 2
  %iv.scaled.add2.div5 = udiv i32 %iv.scaled.add2, 5
  tail call void @bar(i32 %iv.scaled.add2.div5)

  ; root set 2

  ; base instruction
  %iv.scaled.add4 = add nuw nsw i32 %iv.scaled, 4
  %iv.scaled.add4.div5 = udiv i32 %iv.scaled.add4, 5
  tail call void @bar(i32 %iv.scaled.add4.div5)
  ; root instruction 0
  %iv.scaled.add5 = add nuw nsw i32 %iv.scaled, 5
  %iv.scaled.add5.div5 = udiv i32 %iv.scaled.add5, 5
  tail call void @bar(i32 %iv.scaled.add5.div5)
  ; root instruction 2
  %iv.scaled.add6 = add nuw nsw i32 %iv.scaled, 6
  %iv.scaled.add6.div5 = udiv i32 %iv.scaled.add6, 5
  tail call void @bar(i32 %iv.scaled.add6.div5)

  ; loop-increment
  %iv.next = add nuw nsw i32 %iv, 1

  ; latch
  %cmp = icmp ult i32 %iv.next, 3
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define dso_local void @unrerollable3() {
entry:
  br label %loop

loop:

; CHECK-LABEL: loop:
; CHECK-NEXT:    %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
; CHECK-NEXT:    %iv.mul3 = mul nuw nsw i32 %iv, 3
; CHECK-NEXT:    %iv.scaled = add nuw nsw i32 %iv.mul3, 20
; CHECK-NEXT:    %iv.mul7 = mul nuw nsw i32 %iv, 7
; CHECK-NEXT:    tail call void @bar(i32 %iv.mul7)

  ; induction variable
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]

  ; scale instruction
  %iv.mul3 = mul nuw nsw i32 %iv, 3

  ; extra simple arithmetic operations, used by root instructions
  %iv.scaled = add nuw nsw i32 %iv.mul3, 20

  ; unrerollable instructions using %iv
  %iv.mul7 = mul nuw nsw i32 %iv, 7
  tail call void @bar(i32 %iv.mul7)

  ; root set 1

  ; base instruction
  %iv.scaled.div5 = udiv i32 %iv.scaled, 5
  tail call void @bar(i32 %iv.scaled.div5)
  ; root instruction 0
  %iv.scaled.add1 = add nuw nsw i32 %iv.scaled, 1
  %iv.scaled.add1.div5 = udiv i32 %iv.scaled.add1, 5
  tail call void @bar(i32 %iv.scaled.add1.div5)
  ; root instruction 2
  %iv.scaled.add2 = add nuw nsw i32 %iv.scaled, 2
  %iv.scaled.add2.div5 = udiv i32 %iv.scaled.add2, 5
  tail call void @bar(i32 %iv.scaled.add2.div5)

  ; root set 2

  ; base instruction
  %iv.scaled.add4 = add nuw nsw i32 %iv.scaled, 4
  %iv.scaled.add4.div5 = udiv i32 %iv.scaled.add4, 5
  tail call void @bar(i32 %iv.scaled.add4.div5)
  ; root instruction 0
  %iv.scaled.add5 = add nuw nsw i32 %iv.scaled, 5
  %iv.scaled.add5.div5 = udiv i32 %iv.scaled.add5, 5
  tail call void @bar(i32 %iv.scaled.add5.div5)
  ; root instruction 2
  %iv.scaled.add6 = add nuw nsw i32 %iv.scaled, 6
  %iv.scaled.add6.div5 = udiv i32 %iv.scaled.add6, 5
  tail call void @bar(i32 %iv.scaled.add6.div5)

  ; loop-increment
  %iv.next = add nuw nsw i32 %iv, 1

  ; latch
  %cmp = icmp ult i32 %iv.next, 3
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

declare dso_local void @bar(i32)
