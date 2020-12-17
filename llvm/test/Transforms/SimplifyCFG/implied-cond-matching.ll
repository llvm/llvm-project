; RUN: opt %s -S -simplifycfg -simplifycfg-require-and-preserve-domtree=1 | FileCheck %s

declare void @dead()
declare void @alive()
declare void @is(i1)

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

; A == B implies A == B is true.
; CHECK-LABEL: @test_eq_eq
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_eq_eq(i32 %a, i32 %b) {
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp eq i32 %a, %b
  br i1 %cmp2, label %eq_eq_istrue, label %eq_eq_isfalse

eq_eq_istrue:
  call void @is(i1 true)
  ret void

eq_eq_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A == B implies A != B is false.
; CHECK-LABEL: @test_eq_ne
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_eq_ne(i32 %a, i32 %b) {
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp ne i32 %a, %b
  br i1 %cmp2, label %eq_ne_istrue, label %eq_ne_isfalse

eq_ne_istrue:
  call void @is(i1 true)
  ret void

eq_ne_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A == B implies A >u B is false.
; CHECK-LABEL: @test_eq_ugt
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_eq_ugt(i32 %a, i32 %b) {
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp ugt i32 %a, %b
  br i1 %cmp2, label %eq_ugt_istrue, label %eq_ugt_isfalse

eq_ugt_istrue:
  call void @is(i1 true)
  ret void

eq_ugt_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A == B implies A >=u B is true.
; CHECK-LABEL: @test_eq_uge
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_eq_uge(i32 %a, i32 %b) {
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp uge i32 %a, %b
  br i1 %cmp2, label %eq_uge_istrue, label %eq_uge_isfalse

eq_uge_istrue:
  call void @is(i1 true)
  ret void

eq_uge_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A == B implies A <u B is false.
; CHECK-LABEL: @test_eq_ult
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_eq_ult(i32 %a, i32 %b) {
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp ult i32 %a, %b
  br i1 %cmp2, label %eq_ult_istrue, label %eq_ult_isfalse

eq_ult_istrue:
  call void @is(i1 true)
  ret void

eq_ult_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A == B implies A <=u B is true.
; CHECK-LABEL: @test_eq_ule
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_eq_ule(i32 %a, i32 %b) {
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp ule i32 %a, %b
  br i1 %cmp2, label %eq_ule_istrue, label %eq_ule_isfalse

eq_ule_istrue:
  call void @is(i1 true)
  ret void

eq_ule_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A == B implies A >s B is false.
; CHECK-LABEL: @test_eq_sgt
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_eq_sgt(i32 %a, i32 %b) {
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp sgt i32 %a, %b
  br i1 %cmp2, label %eq_sgt_istrue, label %eq_sgt_isfalse

eq_sgt_istrue:
  call void @is(i1 true)
  ret void

eq_sgt_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A == B implies A >=s B is true.
; CHECK-LABEL: @test_eq_sge
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_eq_sge(i32 %a, i32 %b) {
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp sge i32 %a, %b
  br i1 %cmp2, label %eq_sge_istrue, label %eq_sge_isfalse

eq_sge_istrue:
  call void @is(i1 true)
  ret void

eq_sge_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A == B implies A <s B is false.
; CHECK-LABEL: @test_eq_slt
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_eq_slt(i32 %a, i32 %b) {
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp slt i32 %a, %b
  br i1 %cmp2, label %eq_slt_istrue, label %eq_slt_isfalse

eq_slt_istrue:
  call void @is(i1 true)
  ret void

eq_slt_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A == B implies A <=s B is true.
; CHECK-LABEL: @test_eq_sle
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_eq_sle(i32 %a, i32 %b) {
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp sle i32 %a, %b
  br i1 %cmp2, label %eq_sle_istrue, label %eq_sle_isfalse

eq_sle_istrue:
  call void @is(i1 true)
  ret void

eq_sle_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A != B implies A != B is true.
; CHECK-LABEL: @test_ne_ne
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_ne_ne(i32 %a, i32 %b) {
  %cmp1 = icmp ne i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp ne i32 %a, %b
  br i1 %cmp2, label %ne_ne_istrue, label %ne_ne_isfalse

ne_ne_istrue:
  call void @is(i1 true)
  ret void

ne_ne_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A != B implies A >u B is unknown to be true or false.
; CHECK-LABEL: @test_ne_ugt
; CHECK: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_ne_ugt(i32 %a, i32 %b) {
  %cmp1 = icmp ne i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp ugt i32 %a, %b
  br i1 %cmp2, label %ne_ugt_istrue, label %ne_ugt_isfalse

ne_ugt_istrue:
  call void @is(i1 true)
  ret void

ne_ugt_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A != B implies A >=u B is unknown to be true or false.
; CHECK-LABEL: @test_ne_uge
; CHECK: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_ne_uge(i32 %a, i32 %b) {
  %cmp1 = icmp ne i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp uge i32 %a, %b
  br i1 %cmp2, label %ne_uge_istrue, label %ne_uge_isfalse

ne_uge_istrue:
  call void @is(i1 true)
  ret void

ne_uge_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A != B implies A <u B is unknown to be true or false.
; CHECK-LABEL: @test_ne_ult
; CHECK: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_ne_ult(i32 %a, i32 %b) {
  %cmp1 = icmp ne i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp ult i32 %a, %b
  br i1 %cmp2, label %ne_ult_istrue, label %ne_ult_isfalse

ne_ult_istrue:
  call void @is(i1 true)
  ret void

ne_ult_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A != B implies A <=u B is unknown to be true or false.
; CHECK-LABEL: @test_ne_ule
; CHECK: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_ne_ule(i32 %a, i32 %b) {
  %cmp1 = icmp ne i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp ule i32 %a, %b
  br i1 %cmp2, label %ne_ule_istrue, label %ne_ule_isfalse

ne_ule_istrue:
  call void @is(i1 true)
  ret void

ne_ule_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A != B implies A >s B is unknown to be true or false.
; CHECK-LABEL: @test_ne_sgt
; CHECK: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_ne_sgt(i32 %a, i32 %b) {
  %cmp1 = icmp ne i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp sgt i32 %a, %b
  br i1 %cmp2, label %ne_sgt_istrue, label %ne_sgt_isfalse

ne_sgt_istrue:
  call void @is(i1 true)
  ret void

ne_sgt_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A != B implies A >=s B is unknown to be true or false.
; CHECK-LABEL: @test_ne_sge
; CHECK: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_ne_sge(i32 %a, i32 %b) {
  %cmp1 = icmp ne i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp sge i32 %a, %b
  br i1 %cmp2, label %ne_sge_istrue, label %ne_sge_isfalse

ne_sge_istrue:
  call void @is(i1 true)
  ret void

ne_sge_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A != B implies A <s B is unknown to be true or false.
; CHECK-LABEL: @test_ne_slt
; CHECK: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_ne_slt(i32 %a, i32 %b) {
  %cmp1 = icmp ne i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp slt i32 %a, %b
  br i1 %cmp2, label %ne_slt_istrue, label %ne_slt_isfalse

ne_slt_istrue:
  call void @is(i1 true)
  ret void

ne_slt_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A != B implies A <=s B is unknown to be true or false.
; CHECK-LABEL: @test_ne_sle
; CHECK: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_ne_sle(i32 %a, i32 %b) {
  %cmp1 = icmp ne i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp sle i32 %a, %b
  br i1 %cmp2, label %ne_sle_istrue, label %ne_sle_isfalse

ne_sle_istrue:
  call void @is(i1 true)
  ret void

ne_sle_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A >u B implies A >u B is true.
; CHECK-LABEL: @test_ugt_ugt
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_ugt_ugt(i32 %a, i32 %b) {
  %cmp1 = icmp ugt i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp ugt i32 %a, %b
  br i1 %cmp2, label %ugt_ugt_istrue, label %ugt_ugt_isfalse

ugt_ugt_istrue:
  call void @is(i1 true)
  ret void

ugt_ugt_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A >u B implies A >=u B is true.
; CHECK-LABEL: @test_ugt_uge
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_ugt_uge(i32 %a, i32 %b) {
  %cmp1 = icmp ugt i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp uge i32 %a, %b
  br i1 %cmp2, label %ugt_uge_istrue, label %ugt_uge_isfalse

ugt_uge_istrue:
  call void @is(i1 true)
  ret void

ugt_uge_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A >u B implies A <u B is false.
; CHECK-LABEL: @test_ugt_ult
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_ugt_ult(i32 %a, i32 %b) {
  %cmp1 = icmp ugt i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp ult i32 %a, %b
  br i1 %cmp2, label %ugt_ult_istrue, label %ugt_ult_isfalse

ugt_ult_istrue:
  call void @is(i1 true)
  ret void

ugt_ult_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A >u B implies A <=u B is false.
; CHECK-LABEL: @test_ugt_ule
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_ugt_ule(i32 %a, i32 %b) {
  %cmp1 = icmp ugt i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp ule i32 %a, %b
  br i1 %cmp2, label %ugt_ule_istrue, label %ugt_ule_isfalse

ugt_ule_istrue:
  call void @is(i1 true)
  ret void

ugt_ule_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A >=u B implies A >=u B is true.
; CHECK-LABEL: @test_uge_uge
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_uge_uge(i32 %a, i32 %b) {
  %cmp1 = icmp uge i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp uge i32 %a, %b
  br i1 %cmp2, label %uge_uge_istrue, label %uge_uge_isfalse

uge_uge_istrue:
  call void @is(i1 true)
  ret void

uge_uge_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A >=u B implies A <u B is false.
; CHECK-LABEL: @test_uge_ult
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_uge_ult(i32 %a, i32 %b) {
  %cmp1 = icmp uge i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp ult i32 %a, %b
  br i1 %cmp2, label %uge_ult_istrue, label %uge_ult_isfalse

uge_ult_istrue:
  call void @is(i1 true)
  ret void

uge_ult_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A >=u B implies A <=u B is unknown to be true or false.
; CHECK-LABEL: @test_uge_ule
; CHECK: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_uge_ule(i32 %a, i32 %b) {
  %cmp1 = icmp uge i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp ule i32 %a, %b
  br i1 %cmp2, label %uge_ule_istrue, label %uge_ule_isfalse

uge_ule_istrue:
  call void @is(i1 true)
  ret void

uge_ule_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A <u B implies A <u B is true.
; CHECK-LABEL: @test_ult_ult
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_ult_ult(i32 %a, i32 %b) {
  %cmp1 = icmp ult i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp ult i32 %a, %b
  br i1 %cmp2, label %ult_ult_istrue, label %ult_ult_isfalse

ult_ult_istrue:
  call void @is(i1 true)
  ret void

ult_ult_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A <u B implies A <=u B is true.
; CHECK-LABEL: @test_ult_ule
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_ult_ule(i32 %a, i32 %b) {
  %cmp1 = icmp ult i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp ule i32 %a, %b
  br i1 %cmp2, label %ult_ule_istrue, label %ult_ule_isfalse

ult_ule_istrue:
  call void @is(i1 true)
  ret void

ult_ule_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A <=u B implies A <=u B is true.
; CHECK-LABEL: @test_ule_ule
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_ule_ule(i32 %a, i32 %b) {
  %cmp1 = icmp ule i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp ule i32 %a, %b
  br i1 %cmp2, label %ule_ule_istrue, label %ule_ule_isfalse

ule_ule_istrue:
  call void @is(i1 true)
  ret void

ule_ule_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A >s B implies A >s B is true.
; CHECK-LABEL: @test_sgt_sgt
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_sgt_sgt(i32 %a, i32 %b) {
  %cmp1 = icmp sgt i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp sgt i32 %a, %b
  br i1 %cmp2, label %sgt_sgt_istrue, label %sgt_sgt_isfalse

sgt_sgt_istrue:
  call void @is(i1 true)
  ret void

sgt_sgt_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A >s B implies A >=s B is true.
; CHECK-LABEL: @test_sgt_sge
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_sgt_sge(i32 %a, i32 %b) {
  %cmp1 = icmp sgt i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp sge i32 %a, %b
  br i1 %cmp2, label %sgt_sge_istrue, label %sgt_sge_isfalse

sgt_sge_istrue:
  call void @is(i1 true)
  ret void

sgt_sge_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A >s B implies A <s B is false.
; CHECK-LABEL: @test_sgt_slt
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_sgt_slt(i32 %a, i32 %b) {
  %cmp1 = icmp sgt i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp slt i32 %a, %b
  br i1 %cmp2, label %sgt_slt_istrue, label %sgt_slt_isfalse

sgt_slt_istrue:
  call void @is(i1 true)
  ret void

sgt_slt_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A >s B implies A <=s B is false.
; CHECK-LABEL: @test_sgt_sle
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_sgt_sle(i32 %a, i32 %b) {
  %cmp1 = icmp sgt i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp sle i32 %a, %b
  br i1 %cmp2, label %sgt_sle_istrue, label %sgt_sle_isfalse

sgt_sle_istrue:
  call void @is(i1 true)
  ret void

sgt_sle_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A >=s B implies A >=s B is true.
; CHECK-LABEL: @test_sge_sge
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_sge_sge(i32 %a, i32 %b) {
  %cmp1 = icmp sge i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp sge i32 %a, %b
  br i1 %cmp2, label %sge_sge_istrue, label %sge_sge_isfalse

sge_sge_istrue:
  call void @is(i1 true)
  ret void

sge_sge_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A >=s B implies A <s B is false.
; CHECK-LABEL: @test_sge_slt
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_sge_slt(i32 %a, i32 %b) {
  %cmp1 = icmp sge i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp slt i32 %a, %b
  br i1 %cmp2, label %sge_slt_istrue, label %sge_slt_isfalse

sge_slt_istrue:
  call void @is(i1 true)
  ret void

sge_slt_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A >=s B implies A <=s B is unknown to be true or false.
; CHECK-LABEL: @test_sge_sle
; CHECK: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_sge_sle(i32 %a, i32 %b) {
  %cmp1 = icmp sge i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp sle i32 %a, %b
  br i1 %cmp2, label %sge_sle_istrue, label %sge_sle_isfalse

sge_sle_istrue:
  call void @is(i1 true)
  ret void

sge_sle_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A <s B implies A <s B is true.
; CHECK-LABEL: @test_slt_slt
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_slt_slt(i32 %a, i32 %b) {
  %cmp1 = icmp slt i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp slt i32 %a, %b
  br i1 %cmp2, label %slt_slt_istrue, label %slt_slt_isfalse

slt_slt_istrue:
  call void @is(i1 true)
  ret void

slt_slt_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A <s B implies A <=s B is true.
; CHECK-LABEL: @test_slt_sle
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_slt_sle(i32 %a, i32 %b) {
  %cmp1 = icmp slt i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp sle i32 %a, %b
  br i1 %cmp2, label %slt_sle_istrue, label %slt_sle_isfalse

slt_sle_istrue:
  call void @is(i1 true)
  ret void

slt_sle_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A <=s B implies A <=s B is true.
; CHECK-LABEL: @test_sle_sle
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_sle_sle(i32 %a, i32 %b) {
  %cmp1 = icmp sle i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp sle i32 %a, %b
  br i1 %cmp2, label %sle_sle_istrue, label %sle_sle_isfalse

sle_sle_istrue:
  call void @is(i1 true)
  ret void

sle_sle_isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; A >=u 5 implies A <u 5 is false.
; CHECK-LABEL: @test_uge_ult_const
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_uge_ult_const(i32 %a, i32 %b) {
  %cmp1 = icmp uge i32 %a, 5
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp ult i32 %a, 5
  br i1 %cmp2, label %istrue, label %isfalse

istrue:
  call void @is(i1 true)
  ret void

isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}
