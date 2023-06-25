; RUN: opt -passes=constraint-elimination -constraint-elimination-dump-reproducers -pass-remarks=constraint-elimination %s 2>&1 | FileCheck %s

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

declare void @use(i1)

declare void @llvm.assume(i1)

define i1 @test_no_known_facts(ptr %dst) {
; CHECK: remark: <unknown>:0:0: module; ModuleID = 'test_no_known_facts'
; CHECK-LABEL: define i1 @"{{.+}}test_no_known_factsrepro"(ptr %dst)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %dst.0 = getelementptr inbounds ptr, ptr %dst, i64 0
; CHECK-NEXT:    %upper = getelementptr inbounds ptr, ptr %dst, i64 2
; CHECK-NEXT:    %c = icmp ult ptr %dst.0, %upper
; CHECK-NEXT:    ret i1 %c
; CHECK-NEXT:  }
;
entry:
  %dst.0 = getelementptr inbounds ptr, ptr %dst, i64 0
  %upper = getelementptr inbounds ptr, ptr %dst, i64 2
  %c = icmp ult i32* %dst.0, %upper
  ret i1 %c
}

define void @test_one_known_fact_true_branch(i8 %start, i8 %high) {
; CHECK: remark: <unknown>:0:0: module; ModuleID = 'test_one_known_fact_true_branch'

; CHECK-LABEL: define i1 @"{{.*}}test_one_known_fact_true_branchrepro"(i8 %high, i8 %start) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %add.ptr.i = add nuw i8 %start, 3
; CHECK-NEXT:   %0 = icmp ult i8 %add.ptr.i, %high
; CHECK-NEXT:   call void @llvm.assume(i1 %0)
; CHECK-NEXT:   %t.0 = icmp ult i8 %start, %high
; CHECK-NEXT:   ret i1 %t.0
; CHECK-NEXT: }
;
entry:
  %add.ptr.i = add nuw i8 %start, 3
  %c.1 = icmp ult i8 %add.ptr.i, %high
  br i1 %c.1, label %if.then, label %if.end

if.then:
  %t.0 = icmp ult i8 %start, %high
  call void @use(i1 %t.0)
  ret void

if.end:
  ret void
}

define void @test_one_known_fact_false_branch(i8 %start, i8 %high) {
; CHECK: remark: <unknown>:0:0: module; ModuleID = 'test_one_known_fact_false_branch'
;
; CHECK-LABEL:define i1 @"{{.*}}test_one_known_fact_false_branchrepro"(i8 %high, i8 %start) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %add.ptr.i = add nuw i8 %start, 3
; CHECK-NEXT:   %0 = icmp ult i8 %add.ptr.i, %high
; CHECK-NEXT:   call void @llvm.assume(i1 %0)
; CHECK-NEXT:   %t.0 = icmp ult i8 %start, %high
; CHECK-NEXT:   ret i1 %t.0
; CHECK-NEXT: }
;
entry:
  %add.ptr.i = add nuw i8 %start, 3
  %c.1 = icmp uge i8 %add.ptr.i, %high
  br i1 %c.1, label %if.then, label %if.end

if.then:
  ret void

if.end:
  %t.0 = icmp ult i8 %start, %high
  call void @use(i1 %t.0)
  ret void
}

define void @test_multiple_known_facts_branches_1(i8 %a, i8 %b) {
; CHECK: remark: <unknown>:0:0: module; ModuleID = 'test_multiple_known_facts_branches_1'

; CHECK-LABEL: define i1 @"{{.*}}test_multiple_known_facts_branches_1repro"(i8 %a, i8 %b) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = icmp ugt i8 %a, 10
; CHECK-NEXT:   call void @llvm.assume(i1 %0)
; CHECK-NEXT:   %1 = icmp ugt i8 %b, 10
; CHECK-NEXT:   call void @llvm.assume(i1 %1)
; CHECK-NEXT:   %add = add nuw i8 %a, %b
; CHECK-NEXT:   %t.0 = icmp ugt i8 %add, 20
; CHECK-NEXT:   ret i1 %t.0
; CHECK-NEXT: }
;
entry:
  %c.1 = icmp ugt i8 %a, 10
  br i1 %c.1, label %then.1, label %else.1

then.1:
  %c.2 = icmp ugt i8 %b, 10
  br i1 %c.2, label %then.2, label %else.1

then.2:
  %add = add nuw i8 %a, %b
  %t.0 = icmp ugt i8 %add, 20
  call void @use(i1 %t.0)
  ret void

else.1:
  ret void
}

define void @test_multiple_known_facts_branches_2(i8 %a, i8 %b) {
; CHECK: remark: <unknown>:0:0: module; ModuleID = 'test_multiple_known_facts_branches_2'
;
; CHECK-LABEL: define i1 @"{{.*}}test_multiple_known_facts_branches_2repro"(i8 %a, i8 %b) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = icmp ugt i8 %a, 10
; CHECK-NEXT:   call void @llvm.assume(i1 %0)
; CHECK-NEXT:   %1 = icmp ugt i8 %b, 10
; CHECK-NEXT:   call void @llvm.assume(i1 %1)
; CHECK-NEXT:   %add = add nuw i8 %a, %b
; CHECK-NEXT:   %t.0 = icmp ugt i8 %add, 20
; CHECK-NEXT:   ret i1 %t.0
; CHECK-NEXT: }
;
entry:
  %c.1 = icmp ugt i8 %a, 10
  br i1 %c.1, label %then.1, label %exit

then.1:
  %c.2 = icmp ule i8 %b, 10
  br i1 %c.2, label %exit, label %else.2

else.2:
  %add = add nuw i8 %a, %b
  %t.0 = icmp ugt i8 %add, 20
  call void @use(i1 %t.0)
  ret void

exit:
  ret void
}

define void @test_assumes(i8 %a, i8 %b) {
; CHECK-LABEL: define i1 @"{{.*}}test_assumesrepro.2"(i8 %a, i8 %b) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = icmp ugt i8 %a, 10
; CHECK-NEXT:   call void @llvm.assume(i1 %0)
; CHECK-NEXT:   %1 = icmp ugt i8 %b, 10
; CHECK-NEXT:   call void @llvm.assume(i1 %1)
; CHECK-NEXT:   %add = add nuw i8 %a, %b
; CHECK-NEXT:   %t.0 = icmp ult i8 %add, 20
; CHECK-NEXT:   ret i1 %t.0
; CHECK-NEXT: }
;
entry:
  %c.1 = icmp ugt i8 %a, 10
  call void @llvm.assume(i1 %c.1)
  %c.2 = icmp ugt i8 %b, 10
  call void @llvm.assume(i1 %c.2)
  %add = add nuw i8 %a, %b
  %t.0 = icmp ult i8 %add, 20
  call void @use(i1 %t.0)
  ret void
}

declare void @noundef(ptr noundef)

; Currently this fails decomposition. No reproducer should be generated.
define i1 @test_inbounds_precondition(ptr %src, i32 %n, i32 %idx) {
; CHECK-NOT: test_inbounds_precondition
entry:
  %upper = getelementptr inbounds i32, ptr %src, i64 5
  %src.idx.4 = getelementptr i32, ptr %src, i64 4
  %cmp.upper.4 = icmp ule ptr %src.idx.4, %upper
  br i1 %cmp.upper.4, label %then, label %else

then:
  ret i1 true

else:
  ret i1 false
}

define i32 @test_branch(i32 %a) {
; CHECK-LABEL: define i1 @"{{.+}}test_branchrepro"(i32 %a) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = icmp ult i32 %a, 0
; CHECK-NEXT:   call void @llvm.assume(i1 %0)
; CHECK-NEXT:   %c.2 = icmp ugt i32 0, 0
; CHECK-NEXT:   ret i1 %c.2
; CHECK-NEXT: }
;
entry:
  %c.1 = icmp ult i32 %a, 0
  br i1 %c.1, label %then, label %exit

then:
  %c.2 = icmp ugt i32 0, 0
  call void @use(i1 %c.2)
  br label %exit

exit:
  ret i32 0
}

define i32 @test_invoke(i32 %a) personality ptr null {
; CHECK-LABEL: define i1 @"{{.+}}test_invokerepro"(i32 %l, i32 %a) {
; CHECK-NEXT: entry:
; CHECK-NEXT:  %0 = icmp slt i32 %a, %l
; CHECK-NEXT:  call void @llvm.assume(i1 %0)
; CHECK-NEXT:  %c.2 = icmp eq i32 0, 0
; CHECK-NEXT:  ret i1 %c.2
; CHECK-NEXT:}
;
entry:
  %call = invoke ptr null(i64 0)
          to label %cont unwind label %lpad

cont:
  %l = load i32, ptr %call, align 4
  %c.1 = icmp slt i32 %a, %l
  br i1 %c.1, label %then, label %exit

lpad:
  %lp = landingpad { ptr, i32 }
          catch ptr null
          catch ptr null
  ret i32 0

then:
  %c.2 = icmp eq i32 0, 0
  call void @use(i1 %c.2)
  br label %exit

exit:
  ret i32 0
}

define <2 x i1> @vector_cmp(<2 x ptr> %vec) {
; CHECK-LABEL: define <2 x i1> @"{{.+}}vector_cmprepro"(<2 x ptr> %vec) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep.1 = getelementptr inbounds i32, <2 x ptr> %vec, i64 1
; CHECK-NEXT:   %t.1 = icmp ult <2 x ptr> %vec, %gep.1
; CHECK-NEXT:   ret <2 x i1> %t.1
; CHECK-NEXT: }
;
  %gep.1 = getelementptr inbounds i32, <2 x ptr> %vec, i64 1
  %t.1 = icmp ult <2 x ptr> %vec, %gep.1
  ret <2 x i1> %t.1
}

define i1 @shared_operand() {
; CHECK-LABEL: define i1 @"{{.+}}shared_operandrepro"() {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %sub = sub i8 0, 0
; CHECK-NEXT:   %sub.2 = sub nuw i8 %sub, 0
; CHECK-NEXT:   %c.5 = icmp ult i8 %sub.2, %sub
; CHECK-NEXT:   ret i1 %c.5
; CHECK-NEXT: }
;
entry:
  %sub = sub i8 0, 0
  %sub.2 = sub nuw i8 %sub, 0
  %c.5 = icmp ult i8 %sub.2, %sub
  ret i1 %c.5
}

@glob = external global i32

define i1 @load_global() {
; CHECK-LABEL: define i1 @"{{.*}}load_globalrepro"(i32 %l) {
; CHECK-NEXT: entry:
; CHECK-NEXT:  %c = icmp ugt i32 %l, %l
; CHECK-NEXT:  ret i1 %c
; CHECK-NEXT:}
;
entry:
  %l = load i32, ptr @glob, align 8
  %c = icmp ugt i32 %l, %l
  ret i1 %c
}

define i1 @test_ptr_null_constant(ptr %a) {
; CHECK-LABEL: define i1 @"{{.+}}test_ptr_null_constantrepro"(ptr %a) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = icmp eq ptr %a, null
; CHECK-NEXT:   call void @llvm.assume(i1 %0)
; CHECK-NEXT:   %c.2 = icmp eq ptr %a, null
; CHECK-NEXT:   ret i1 %c.2
; CHECK-NEXT: }
;
entry:
  %c.1 = icmp eq ptr %a, null
  br i1 %c.1, label %then, label %else

then:
  %c.2 = icmp eq ptr %a, null
  ret i1 %c.2

else:
  ret i1 false
}

define i1 @test_both_signed_and_unsigned_conds_needed_in_reproducer(ptr %src, ptr %lower, ptr %upper, i16 %N) {
; CHECK-LABEL: define i1 @"{{.+}}test_both_signed_and_unsigned_conds_needed_in_reproducerrepro"(i16 %N, ptr %src) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = icmp sge i16 %N, 0
; CHECK-NEXT:   call void @llvm.assume(i1 %0)
; CHECK-NEXT:   %src.end = getelementptr inbounds i8, ptr %src, i16 %N
; CHECK-NEXT:   %cmp.src.start = icmp ule ptr %src, %src.end
; CHECK-NEXT:   ret i1 %cmp.src.start
; CHECK-NEXT: }
;
entry:
  %N.pos = icmp sge i16 %N, 0
  br i1 %N.pos, label %then, label %else

then:
  %src.end = getelementptr inbounds i8, ptr %src, i16 %N
  %cmp.src.start = icmp ule ptr %src, %src.end
  ret i1 %cmp.src.start

else:
  ret i1 false
}
