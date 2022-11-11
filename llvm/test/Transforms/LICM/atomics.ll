; RUN: opt -aa-pipeline=basic-aa -passes='require<opt-remark-emit>,loop-mssa(licm)' < %s -S | FileCheck %s

; Check that we can hoist unordered loads
define i32 @test1(ptr nocapture %y) nounwind uwtable ssp {
entry:
  br label %loop

loop:
  %i = phi i32 [ %inc, %loop ], [ 0, %entry ]
  %val = load atomic i32, ptr %y unordered, align 4
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
define i32 @test2(ptr nocapture %y) nounwind uwtable ssp {
entry:
  br label %loop

loop:
  %val = load atomic i32, ptr %y monotonic, align 4
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
define i32 @test3(ptr nocapture noalias %x, ptr nocapture %y) nounwind uwtable ssp {
entry:
  br label %loop

loop:
  %vala = load atomic i32, ptr %y monotonic, align 4
  %valb = load atomic i32, ptr %x unordered, align 4
  %exitcond = icmp ne i32 %vala, %valb
  br i1 %exitcond, label %end, label %loop

end:
  ret i32 %vala
; CHECK-LABEL: define i32 @test3(
; CHECK: load atomic i32, ptr %x unordered
; CHECK-NEXT: br label %loop
}

; We can sink an unordered store
define i32 @test4(ptr nocapture noalias %x, ptr nocapture %y) nounwind uwtable ssp {
entry:
  br label %loop

loop:
  %vala = load atomic i32, ptr %y monotonic, align 4
  store atomic i32 %vala, ptr %x unordered, align 4
  %exitcond = icmp ne i32 %vala, 0
  br i1 %exitcond, label %end, label %loop

end:
  ret i32 %vala
; CHECK-LABEL: define i32 @test4(
; CHECK-LABEL: loop:
; CHECK: load atomic i32, ptr %y monotonic
; CHECK-NOT: store
; CHECK-LABEL: end:
; CHECK-NEXT:   %[[LCSSAPHI:.*]] = phi i32 [ %vala
; CHECK:   store atomic i32 %[[LCSSAPHI]], ptr %x unordered, align 4
}

; We currently don't handle ordered atomics.
define i32 @test5(ptr nocapture noalias %x, ptr nocapture %y) nounwind uwtable ssp {
entry:
  br label %loop

loop:
  %vala = load atomic i32, ptr %y monotonic, align 4
  store atomic i32 %vala, ptr %x release, align 4
  %exitcond = icmp ne i32 %vala, 0
  br i1 %exitcond, label %end, label %loop

end:
  ret i32 %vala
; CHECK-LABEL: define i32 @test5(
; CHECK: load atomic i32, ptr %y monotonic
; CHECK-NEXT: store atomic
}

; We currently don't touch volatiles
define i32 @test6(ptr nocapture noalias %x, ptr nocapture %y) nounwind uwtable ssp {
entry:
  br label %loop

loop:
  %vala = load atomic i32, ptr %y monotonic, align 4
  store volatile i32 %vala, ptr %x, align 4
  %exitcond = icmp ne i32 %vala, 0
  br i1 %exitcond, label %end, label %loop

end:
  ret i32 %vala
; CHECK-LABEL: define i32 @test6(
; CHECK: load atomic i32, ptr %y monotonic
; CHECK-NEXT: store volatile
}

; We currently don't touch volatiles
define i32 @test6b(ptr nocapture noalias %x, ptr nocapture %y) nounwind uwtable ssp {
entry:
  br label %loop

loop:
  %vala = load atomic i32, ptr %y monotonic, align 4
  store atomic volatile i32 %vala, ptr %x unordered, align 4
  %exitcond = icmp ne i32 %vala, 0
  br i1 %exitcond, label %end, label %loop

end:
  ret i32 %vala
; CHECK-LABEL: define i32 @test6b(
; CHECK: load atomic i32, ptr %y monotonic
; CHECK-NEXT: store atomic volatile
}

; Mixing unorder atomics and normal loads/stores is
; current unimplemented
define i32 @test7(ptr nocapture noalias %x, ptr nocapture %y) nounwind uwtable ssp {
entry:
  br label %loop

loop:
  store i32 5, ptr %x
  %vala = load atomic i32, ptr %y monotonic, align 4
  store atomic i32 %vala, ptr %x unordered, align 4
  %exitcond = icmp ne i32 %vala, 0
  br i1 %exitcond, label %end, label %loop

end:
  ret i32 %vala
; CHECK-LABEL: define i32 @test7(
; CHECK: store i32 5, ptr %x
; CHECK-NEXT: load atomic i32, ptr %y
; CHECK-NEXT: store atomic i32
}

; Three provably noalias locations - we can sink normal and unordered, but
;  not monotonic
define i32 @test7b(ptr nocapture noalias %x, ptr nocapture %y, ptr noalias nocapture %z) nounwind uwtable ssp {
entry:
  br label %loop

loop:
  store i32 5, ptr %x
  %vala = load atomic i32, ptr %y monotonic, align 4
  store atomic i32 %vala, ptr %z unordered, align 4
  %exitcond = icmp ne i32 %vala, 0
  br i1 %exitcond, label %end, label %loop

end:
  ret i32 %vala
; CHECK-LABEL: define i32 @test7b(
; CHECK-LABEL: loop:
; CHECK: load atomic i32, ptr %y monotonic
; CHECK-LABEL: end:
; CHECK: store i32 5, ptr %x
; CHECK: store atomic i32 %{{.+}}, ptr %z unordered, align 4
}


define i32 @test8(ptr nocapture noalias %x, ptr nocapture %y) {
entry:
  br label %loop

loop:
  %vala = load atomic i32, ptr %y monotonic, align 4
  store atomic i32 %vala, ptr %x unordered, align 4
  fence release
  %exitcond = icmp ne i32 %vala, 0
  br i1 %exitcond, label %end, label %loop

end:
  ret i32 %vala
; CHECK-LABEL: define i32 @test8(
; CHECK-LABEL: loop:
; CHECK: load atomic i32, ptr %y monotonic
; CHECK-NEXT: store atomic
; CHECK-NEXT: fence
}

; Exact semantics of monotonic accesses are a bit vague in the C++ spec,
; for the moment, be conservative and don't touch them.
define i32 @test9(ptr nocapture noalias %x, ptr nocapture %y) {
entry:
  br label %loop

loop:
  %vala = load atomic i32, ptr %y monotonic, align 4
  store atomic i32 %vala, ptr %x monotonic, align 4
  %exitcond = icmp ne i32 %vala, 0
  br i1 %exitcond, label %end, label %loop

end:
  ret i32 %vala
; CHECK-LABEL: define i32 @test9(
; CHECK-LABEL: loop:
; CHECK: load atomic i32, ptr %y monotonic
; CHECK-NEXT:   store atomic i32 %vala, ptr %x monotonic, align 4
}
