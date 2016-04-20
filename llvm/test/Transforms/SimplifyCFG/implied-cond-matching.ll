; RUN: opt %s -S -simplifycfg | FileCheck %s

; Test same condition with swapped operands.
; void test_swapped_ops(unsigned a, unsigned b) {
;   if (a > b) {
;     if (b > a) <- always false
;       dead();
;     alive();
;   }
; }
;
; CHECK-LABEL: @test_swapped_ops
; CHECK-NOT: call void @dead()
; CHECK: call void @alive()
; CHECK: ret
define void @test_swapped_ops(i32 %a, i32 %b) {
entry:
  %cmp = icmp ugt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end3

if.then:
  %cmp1 = icmp ugt i32 %b, %a
  br i1 %cmp1, label %if.then2, label %if.end

if.then2:
  call void @dead()
  br label %if.end

if.end:
  call void @alive()
  br label %if.end3

if.end3:
  ret void
}

; void test_swapped_pred(unsigned a, unsigned b) {
;   if (a > b) {
;     alive();
;     if (b < a) <- always true; remove branch
;       alive();
;   }
; }
;
; CHECK-LABEL: @test_swapped_pred
; CHECK: call void @alive()
; CHECK-NEXT: call void @alive()
; CHECK: ret
define void @test_swapped_pred(i32 %a, i32 %b) {
entry:
  %cmp = icmp ugt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end3

if.then:
  call void @alive()
  %cmp1 = icmp ult i32 %b, %a
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:
  call void @alive()
  br label %if.end3

if.end3:
  ret void
}

; A == B implies A >u B is false
; A == B implies A <u B is false
; CHECK-LABEL: @test_eq_unsigned
; CHECK-NOT: dead
; CHECK: ret void
define void @test_eq_unsigned(i32 %a, i32 %b) {
entry:
  %cmp = icmp eq i32 %a, %b
  br i1 %cmp, label %if.end, label %if.end9

if.end:
  %cmp3 = icmp ugt i32 %a, %b
  br i1 %cmp3, label %if.then4, label %if.end5

if.then4:
  call void @dead()
  br label %if.end5

if.end5:
  %cmp6 = icmp ult i32 %a, %b
  br i1 %cmp6, label %if.then7, label %if.end8

if.then7:
  call void @dead()
  br label %if.end8

if.end8:
  br label %if.end9

if.end9:
  ret void
}

; A == B implies A >s B is false
; A == B implies A <s B is false
; CHECK-LABEL: @test_eq_signed
; CHECK-NOT: dead
; CHECK: ret void
define void @test_eq_signed(i32 %a, i32 %b) {
entry:
  %cmp = icmp eq i32 %a, %b
  br i1 %cmp, label %if.end, label %if.end9

if.end:
  %cmp3 = icmp sgt i32 %a, %b
  br i1 %cmp3, label %if.then4, label %if.end5

if.then4:
  call void @dead()
  br label %if.end5

if.end5:
  %cmp6 = icmp slt i32 %a, %b
  br i1 %cmp6, label %if.then7, label %if.end8

if.then7:
  call void @dead()
  br label %if.end8

if.end8:
  br label %if.end9

if.end9:
  ret void
}

; A != B implies A == B is false
; CHECK-LABEL: @test_ne
; CHECK-NOT: dead
; CHECK: ret void
define void @test_ne(i32 %a, i32 %b) {
entry:
  %cmp = icmp ne i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end3

if.then:
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end

if.then2:
  call void @dead()
  br label %if.end

if.end:
  br label %if.end3

if.end3:
  ret void
}

; A >u B implies A == B is false.
; A >u B implies A <=u B is false.
; A >u B implies A != B is true.
; CHECK-LABEL: @test_ugt
; CHECK-NOT: dead
; CHECK: alive
; CHECK-NOT: dead
; CHECK: ret void
define void @test_ugt(i32 %a, i32 %b) {
entry:
  %cmp = icmp ugt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end9

if.then:
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end

if.then2:
  call void @dead()
  br label %if.end

if.end:
  %cmp3 = icmp ule i32 %a, %b
  br i1 %cmp3, label %if.then4, label %if.end5

if.then4:
  call void @dead()
  br label %if.end5

if.end5:
  %cmp6 = icmp ne i32 %a, %b
  br i1 %cmp6, label %if.then7, label %if.end8

if.then7:
  call void @alive()
  br label %if.end8

if.end8:
  br label %if.end9

if.end9:
  ret void
}

; A >s B implies A == B is false.
; A >s B implies A <=s B is false.
; A >s B implies A != B is true.
; CHECK-LABEL: @test_sgt
; CHECK-NOT: dead
; CHECK: alive
; CHECK-NOT: dead
; CHECK: ret void
define void @test_sgt(i32 %a, i32 %b) {
entry:
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end9

if.then:
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end

if.then2:
  call void @dead()
  br label %if.end

if.end:
  %cmp3 = icmp sle i32 %a, %b
  br i1 %cmp3, label %if.then4, label %if.end5

if.then4:
  call void @dead()
  br label %if.end5

if.end5:
  %cmp6 = icmp ne i32 %a, %b
  br i1 %cmp6, label %if.then7, label %if.end8

if.then7:
  call void @alive()
  br label %if.end8

if.end8:
  br label %if.end9

if.end9:
  ret void
}

; A <u B implies A == B is false.
; A <u B implies A >=u B is false.
; A <u B implies A != B is true.
; CHECK-LABEL: @test_ult
; CHECK-NOT: dead
; CHECK: alive
; CHECK-NOT: dead
; CHECK: ret void
define void @test_ult(i32 %a, i32 %b) {
entry:
  %cmp = icmp ult i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end9

if.then:
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end

if.then2:
  call void @dead()
  br label %if.end

if.end:
  %cmp3 = icmp uge i32 %a, %b
  br i1 %cmp3, label %if.then4, label %if.end5

if.then4:
  call void @dead()
  br label %if.end5

if.end5:
  %cmp6 = icmp eq i32 %a, %b
  br i1 %cmp6, label %if.end8, label %if.then7

if.then7:
  call void @alive()
  br label %if.end8

if.end8:
  br label %if.end9

if.end9:
  ret void
}

; A <s B implies A == B is false.
; A <s B implies A >=s B is false.
; A <s B implies A != B is true.
; CHECK-LABEL: @test_slt
; CHECK-NOT: dead
; CHECK: alive
; CHECK-NOT: dead
; CHECK: ret void
define void @test_slt(i32 %a, i32 %b) {
entry:
  %cmp = icmp slt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end9

if.then:
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end

if.then2:
  call void @dead()
  br label %if.end

if.end:
  %cmp3 = icmp sge i32 %a, %b
  br i1 %cmp3, label %if.then4, label %if.end5

if.then4:
  call void @dead()
  br label %if.end5

if.end5:
  %cmp6 = icmp eq i32 %a, %b
  br i1 %cmp6, label %if.end8, label %if.then7

if.then7:
  call void @alive()
  br label %if.end8

if.end8:
  br label %if.end9

if.end9:
  ret void
}

; A >=u B implies A <u B is false.
; CHECK-LABEL: @test_uge
; CHECK-NOT: dead
; CHECK: ret void
define void @test_uge(i32 %a, i32 %b) {
entry:
  %cmp = icmp uge i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end3

if.then:
  %cmp1 = icmp ult i32 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end

if.then2:
  call void @dead()
  br label %if.end

if.end:
  br label %if.end3

if.end3:
  ret void
}

; A >=s B implies A <s B is false.
; CHECK-LABEL: @test_sge
; CHECK-NOT: dead
; CHECK: ret void
define void @test_sge(i32 %a, i32 %b) {
entry:
  %cmp = icmp sge i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end3

if.then:
  %cmp1 = icmp slt i32 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end

if.then2:
  call void @dead()
  br label %if.end

if.end:
  br label %if.end3

if.end3:
  ret void
}

; A <=u B implies A >u B is false.
; CHECK-LABEL: @test_ule
; CHECK-NOT: dead
; CHECK: ret void
define void @test_ule(i32 %a, i32 %b) {
entry:
  %cmp = icmp ule i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end3

if.then:
  %cmp1 = icmp ugt i32 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end

if.then2:
  call void @dead()
  br label %if.end

if.end:
  br label %if.end3

if.end3:
  ret void
}

; A <=s B implies A >s B is false.
; CHECK-LABEL: @test_sle
; CHECK-NOT: dead
; CHECK: ret void
define void @test_sle(i32 %a, i32 %b) {
entry:
  %cmp = icmp sle i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end3

if.then:
  %cmp1 = icmp sgt i32 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end

if.then2:
  call void @dead()
  br label %if.end

if.end:
  br label %if.end3

if.end3:
  ret void
}

; A u<= B does not imply A s> B is false.
; CHECK-LABEL: @test_ule_sgt_unsafe
; CHECK: alive
; CHECK: ret void
define void @test_ule_sgt_unsafe(i32 %a, i32 %b) {
entry:
  %cmp = icmp ule i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end3

if.then:
  %cmp1 = icmp sgt i32 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end

if.then2:
  call void @alive()
  br label %if.end

if.end:
  br label %if.end3

if.end3:
  ret void
}

; A u> B does not imply B s> A is false.
; CHECK-LABEL: @test_ugt_sgt_unsafe
; CHECK: alive
; CHECK: ret void
define void @test_ugt_sgt_unsafe(i32 %a, i32 %b) {
entry:
  %cmp = icmp ugt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end3

if.then:
  %cmp1 = icmp sgt i32 %b, %a
  br i1 %cmp1, label %if.then2, label %if.end

if.then2:
  call void @alive()
  br label %if.end

if.end:
  br label %if.end3

if.end3:
  ret void
}

declare void @dead()
declare void @alive()
