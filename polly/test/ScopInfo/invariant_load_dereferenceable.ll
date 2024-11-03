; RUN: opt %loadPolly -polly-print-detect -polly-print-scops \
; RUN: -polly-invariant-load-hoisting=true \
; RUN: -disable-output < %s | FileCheck %s

; CHECK-NOT: Function: foo_undereferanceable

; CHECK:       Function: foo_dereferanceable

; CHECK:       Invariant Accesses: {
; CHECK-NEXT:               ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                   [sizeA] -> { Stmt_for_body_j__TO__for_latch_j[i0, i1] -> MemRef_sizeA_ptr[0] };
; CHECK-NEXT:               Execution Context: [sizeA] -> {  :  }
; CHECK-NEXT:       }

; CHECK:            MayWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:               [sizeA] -> { Stmt_for_body_j__TO__for_latch_j[i0, i1] -> MemRef_A[i1, i0] };

; CHECK-NOT: Function: foo_undereferanceable

define void @foo_dereferanceable(ptr %A, ptr %B, ptr dereferenceable(8) align 8 %sizeA_ptr,
		i32 %lb.i, i32 %lb.j, i32 %ub.i, i32 %ub.j) {
entry:
	br label %for.i

for.i:
	%indvar.i = phi i32 [0, %entry], [%indvar.next.i, %for.latch.i]
	%indvar.next.i = add i32 %indvar.i, 1
	%cmp.i = icmp sle i32 %indvar.i, 1024
	br i1 %cmp.i, label %for.body.i, label %exit

for.body.i:
	br label %for.j

for.j:
	%indvar.j = phi i32 [0, %for.body.i], [%indvar.next.j, %for.latch.j]
	%indvar.next.j = add i32 %indvar.j, 1
	%cmp.j = icmp sle i32 %indvar.j, 1024
	br i1 %cmp.j, label %for.body.j, label %for.latch.i

for.body.j:
	%prod = mul i32 %indvar.j, %indvar.j
	%cmp = icmp sle i32 %prod, 1024
	br i1 %cmp, label %stmt, label %for.latch.j

stmt:
	%sext.i = sext i32 %indvar.i to i64
	%sext.j = sext i32 %indvar.j to i64

	%sizeA = load i64, ptr %sizeA_ptr
	%prodA = mul i64 %sext.j, %sizeA
	%offsetA = add i64 %sext.i, %prodA
	%ptrA = getelementptr double, ptr %A, i64 %offsetA
	store double 42.0, ptr %ptrA

	br label %for.latch.j

for.latch.j:
	br label %for.j

for.latch.i:
	br label %for.i

exit:
	ret void
}

define void @foo_undereferanceable(ptr %A, ptr %B, ptr %sizeA_ptr) {
entry:
	br label %for.i

for.i:
	%indvar.i = phi i32 [0, %entry], [%indvar.next.i, %for.latch.i]
	%indvar.next.i = add i32 %indvar.i, 1
	%cmp.i = icmp sle i32 %indvar.i, 1024
	br i1 %cmp.i, label %for.body.i, label %exit

for.body.i:
	br label %for.j

for.j:
	%indvar.j = phi i32 [0, %for.body.i], [%indvar.next.j, %for.latch.j]
	%indvar.next.j = add i32 %indvar.j, 1
	%cmp.j = icmp sle i32 %indvar.j, 1024
	br i1 %cmp.j, label %for.body.j, label %for.latch.i

for.body.j:
	%prod = mul i32 %indvar.j, %indvar.j
	%cmp = icmp sle i32 %prod, 1024
	br i1 %cmp, label %stmt, label %for.latch.j

stmt:
	%sext.i = sext i32 %indvar.i to i64
	%sext.j = sext i32 %indvar.j to i64

	%sizeA = load i64, ptr %sizeA_ptr
	%prodA = mul i64 %sext.j, %sizeA
	%offsetA = add i64 %sext.i, %prodA
	%ptrA = getelementptr double, ptr %A, i64 %offsetA
	store double 42.0, ptr %ptrA

	br label %for.latch.j

for.latch.j:
	br label %for.j

for.latch.i:
	br label %for.i

exit:
	ret void
}

