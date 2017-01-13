; RUN: opt < %s -S -basicaa -licm | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='lcssa,require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop(licm)' < %s -S | FileCheck %s

; Check that we can hoist unordered loads
define i32 @test1(i32* nocapture %y) nounwind uwtable ssp {
entry:
  br label %loop

loop:
  %i = phi i32 [ %inc, %loop ], [ 0, %entry ]
  %val = load atomic i32, i32* %y unordered, align 4
  %inc = add nsw i32 %i, 1
  %exitcond = icmp eq i32 %inc, %val
  br i1 %exitcond, label %end, label %loop

end:
  ret i32 %val
; CHECK-LABEL: define i32 @test1(
; CHECK: load atomic
; CHECK-NEXT: br label %loop
}

; Check that we don't sink/hoist monotonic loads
; (Strictly speaking, it's not forbidden, but it's supposed to be possible to
; use monotonic for spinlock-like constructs.)
define i32 @test2(i32* nocapture %y) nounwind uwtable ssp {
entry:
  br label %loop

loop:
  %val = load atomic i32, i32* %y monotonic, align 4
  %exitcond = icmp ne i32 %val, 0
  br i1 %exitcond, label %end, label %loop

end:
  ret i32 %val
; CHECK-LABEL: define i32 @test2(
; CHECK: load atomic
; CHECK-NEXT: %exitcond = icmp ne
; CHECK-NEXT: br i1 %exitcond, label %end, label %loop
}

; Check that we hoist unordered around monotonic.
; (The noalias shouldn't be necessary in theory, but LICM isn't quite that
; smart yet.)
define i32 @test3(i32* nocapture noalias %x, i32* nocapture %y) nounwind uwtable ssp {
entry:
  br label %loop

loop:
  %vala = load atomic i32, i32* %y monotonic, align 4
  %valb = load atomic i32, i32* %x unordered, align 4
  %exitcond = icmp ne i32 %vala, %valb
  br i1 %exitcond, label %end, label %loop

end:
  ret i32 %vala
; CHECK-LABEL: define i32 @test3(
; CHECK: load atomic i32, i32* %x unordered
; CHECK-NEXT: br label %loop
}

; Don't try to "sink" unordered stores yet; it is legal, but the machinery
; isn't there.
define i32 @test4(i32* nocapture noalias %x, i32* nocapture %y) nounwind uwtable ssp {
entry:
  br label %loop

loop:
  %vala = load atomic i32, i32* %y monotonic, align 4
  store atomic i32 %vala, i32* %x unordered, align 4
  %exitcond = icmp ne i32 %vala, 0
  br i1 %exitcond, label %end, label %loop

end:
  ret i32 %vala
; CHECK-LABEL: define i32 @test4(
; CHECK: load atomic i32, i32* %y monotonic
; CHECK-NEXT: store atomic
}
