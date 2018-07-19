; RUN: opt -S -jump-threading %s | FileCheck %s

; Value of predicate known on all inputs (trivial case)
; Note: InstCombine/EarlyCSE would also get this case
define void @test(i8* %p, i8** %addr) {
; CHECK-LABEL: @test
entry:
  %cmp0 = icmp eq i8* %p, null
  br i1 %cmp0, label %exit, label %loop
loop:
; CHECK-LABEL: loop:
; CHECK-NEXT: phi
; CHECK-NEXT: br label %loop
  %p1 = phi i8* [%p, %entry], [%p1, %loop]
  %cmp1 = icmp eq i8* %p1, null
  br i1 %cmp1, label %exit, label %loop
exit:
  ret void
}

; Value of predicate known on all inputs (non-trivial)
define void @test2(i8* %p) {
; CHECK-LABEL: @test2
entry:
  %cmp0 = icmp eq i8* %p, null
  br i1 %cmp0, label %exit, label %loop
loop:
  %p1 = phi i8* [%p, %entry], [%p2, %backedge]
  %cmp1 = icmp eq i8* %p1, null
  br i1 %cmp1, label %exit, label %backedge
backedge:
; CHECK-LABEL: backedge:
; CHECK-NEXT: phi
; CHECK-NEXT: bitcast
; CHECK-NEXT: load
; CHECK-NEXT: cmp
; CHECK-NEXT: br 
; CHECK-DAG: label %backedge
  %addr = bitcast i8* %p1 to i8**
  %p2 = load i8*, i8** %addr
  %cmp2 = icmp eq i8* %p2, null
  br i1 %cmp2, label %exit, label %loop
exit:
  ret void
}

; If the inputs don't branch the same way, we can't rewrite
; Well, we could unroll this loop exactly twice, but that's
; a different transform.
define void @test_mixed(i8* %p) {
; CHECK-LABEL: @test_mixed
entry:
  %cmp0 = icmp eq i8* %p, null
  br i1 %cmp0, label %exit, label %loop
loop:
; CHECK-LABEL: loop:
; CHECK-NEXT: phi
; CHECK-NEXT: %cmp1 = icmp
; CHECK-NEXT: br i1 %cmp1
  %p1 = phi i8* [%p, %entry], [%p1, %loop]
  %cmp1 = icmp ne i8* %p1, null
  br i1 %cmp1, label %exit, label %loop
exit:
  ret void
}

