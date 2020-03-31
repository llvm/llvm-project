; RUN: opt < %s -S -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 | FileCheck %s

; Check that we can vectorize this loop without crashing.

; CHECK-LABEL: define {{.*}} @widget(
; CHECK: [[vecInd:%.*]] = phi <4 x i8> [ <i8 0, i8 1, i8 2, i8 3>
; CHECK-NEXT: add <4 x i8> [[vecInd]], <i8 1, i8 1, i8 1, i8 1>

define i8 @widget(i8* %arr, i8 %t9) {
bb:
  br label %bb6

bb6:
  %t1.0 = phi i8* [ %arr, %bb ], [ null, %bb6 ]
  %c = call i1 @cond()
  br i1 %c, label %for.preheader, label %bb6

for.preheader:
  br label %for.body

for.body:
  %iv = phi i8 [ %iv.next, %for.body ], [ 0, %for.preheader ]
  %iv.next = add i8 %iv, 1
  %ptr = getelementptr inbounds i8, i8* %arr, i8 %iv.next
  %t3.i = icmp slt i8 %iv.next, %t9
  %t3.i8 = zext i1 %t3.i to i8
  store i8 %t3.i8, i8* %ptr
  %ec = icmp eq i8* %t1.0, %ptr
  br i1 %ec, label %for.exit, label %for.body

for.exit:
  %iv.next.lcssa = phi i8 [ %iv.next, %for.body ]
  ret i8 %iv.next.lcssa
}

declare i1 @cond()
