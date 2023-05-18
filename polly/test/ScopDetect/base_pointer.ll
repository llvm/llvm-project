; RUN: opt %loadPolly -disable-basic-aa -polly-invariant-load-hoisting=true -polly-print-detect -disable-output < %s | FileCheck %s


target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @base_pointer_in_condition(ptr noalias %A_ptr, i64 %N) nounwind {
entry:
  fence seq_cst
  br label %pre

pre:
  %A = load ptr, ptr %A_ptr
  br i1 true, label %for.i, label %then

for.i:
  %indvar = phi i64 [ 0, %pre ], [ %indvar.next, %for.i ]
  %scevgep = getelementptr i64, ptr %A, i64 %indvar
  store i64 %indvar, ptr %scevgep
  %indvar.next = add nsw i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %N
  br i1 %exitcond, label %then, label %for.i

then:
  br label %return

return:
  fence seq_cst
  ret void
}

; CHECK-LABEL: base_pointer_in_condition
; CHECK: Valid Region for Scop: pre => return

define void @base_pointer_is_argument(ptr %A, i64 %n) {
entry:
  br label %for.i

for.i:
  %indvar.i = phi i64 [ %indvar.i.next, %for.i.inc ], [ 0, %entry ]
  br label %S1

S1:
  %conv = sitofp i64 %indvar.i to float
  %arrayidx5 = getelementptr float, ptr %A, i64 %indvar.i
  store float %conv, ptr %arrayidx5, align 4
  br label %for.i.inc

for.i.inc:
  %indvar.i.next = add i64 %indvar.i, 1
  %exitcond.i = icmp ne i64 %indvar.i.next, %n
  br i1 %exitcond.i, label %for.i, label %exit

exit:
  ret void
}

; CHECK-LABEL: base_pointer_is_argument
; CHECK: Valid Region for Scop: for.i => exit

define void @base_pointer_is_const_expr(i64 %n) {
entry:
  br label %for.i

for.i:
  %indvar.i = phi i64 [ %indvar.i.next, %for.i.inc ], [ 0, %entry ]
  br label %S1

S1:
  %conv = sitofp i64 %indvar.i to float
  %arrayidx5 = getelementptr float, ptr inttoptr (i64 100 to ptr), i64 %indvar.i
  store float %conv, ptr %arrayidx5, align 4
  br label %for.i.inc

for.i.inc:
  %indvar.i.next = add i64 %indvar.i, 1
  %exitcond.i = icmp ne i64 %indvar.i.next, %n
  br i1 %exitcond.i, label %for.i, label %exit

exit:
  ret void
}

; CHECK-LABEL: base_pointer_is_const_expr
; CHECK-LABEL: Valid Region for Scop: for.i => exit

@A = external global float

define void @base_pointer_is_global(i64 %n) {
entry:
  br label %for.i

for.i:
  %indvar.i = phi i64 [ %indvar.i.next, %for.i.inc ], [ 0, %entry ]
  br label %S1

S1:
  %conv = sitofp i64 %indvar.i to float
  %arrayidx5 = getelementptr float, ptr @A, i64 %indvar.i
  store float %conv, ptr %arrayidx5, align 4
  br label %for.i.inc

for.i.inc:
  %indvar.i.next = add i64 %indvar.i, 1
  %exitcond.i = icmp ne i64 %indvar.i.next, %n
  br i1 %exitcond.i, label %for.i, label %exit

exit:
  ret void
}

; CHECK-LABEL: base_pointer_is_global
; CHECK: Valid Region for Scop: for.i => exit

declare ptr @foo()

define void @base_pointer_is_inst_outside(i64 %n) {
entry:
  %A = call ptr @foo()
  br label %for.i

for.i:
  %indvar.i = phi i64 [ %indvar.i.next, %for.i.inc ], [ 0, %entry ]
  br label %S1

S1:
  %conv = sitofp i64 %indvar.i to float
  %arrayidx5 = getelementptr float, ptr %A, i64 %indvar.i
  store float %conv, ptr %arrayidx5, align 4
  br label %for.i.inc

for.i.inc:
  %indvar.i.next = add i64 %indvar.i, 1
  %exitcond.i = icmp ne i64 %indvar.i.next, %n
  br i1 %exitcond.i, label %for.i, label %exit

exit:
  ret void
}

; CHECK-LABEL: base_pointer_is_inst_outside
; CHECK: Valid Region for Scop: for.i => exit

declare ptr @getNextBasePtr(ptr) readnone nounwind

define void @base_pointer_is_phi_node(i64 %n, ptr %A) {
entry:
  br label %for.i

for.i:
  %indvar.i = phi i64 [ %indvar.i.next, %for.i.inc ], [ 0, %entry ]
  %ptr = phi ptr [ %ptr.next, %for.i.inc ], [ %A, %entry ]
; To get a PHI node inside a SCoP that can not be analyzed but
; for which the surrounding SCoP is normally still valid we use a function
; without any side effects.
  %ptr.next = call ptr @getNextBasePtr(ptr %ptr)
  br label %S1

S1:
  %conv = sitofp i64 %indvar.i to float
  %arrayidx5 = getelementptr float, ptr %ptr, i64 %indvar.i
  store float %conv, ptr %arrayidx5, align 4
  br label %for.i.inc

for.i.inc:
  %indvar.i.next = add i64 %indvar.i, 1
  %exitcond.i = icmp ne i64 %indvar.i.next, %n
  br i1 %exitcond.i, label %for.i, label %exit

exit:
  ret void
}

; CHECK-LABEL: base_pointer_is_phi_node
; CHECK-NOT: Valid Region for Scop

define void @base_pointer_is_inst_inside_invariant_1(i64 %n, ptr %A) {
entry:
  br label %for.i

for.i:
  %indvar.i = phi i64 [ %indvar.i.next, %for.i.inc ], [ 0, %entry ]
; A function return value, even with readnone nounwind attributes, is not
; considered a valid base pointer because it can return a pointer that aliases
; with something else (e.g. %A or a global) or return a different pointer at
; every call (e.g. malloc)
  %ptr = call ptr @getNextBasePtr(ptr %A)
  br label %S1

S1:
  %conv = sitofp i64 %indvar.i to float
  %arrayidx5 = getelementptr float, ptr %ptr, i64 %indvar.i
  store float %conv, ptr %arrayidx5, align 4
  br label %for.i.inc

for.i.inc:
  %indvar.i.next = add i64 %indvar.i, 1
  %exitcond.i = icmp ne i64 %indvar.i.next, %n
  br i1 %exitcond.i, label %for.i, label %exit

exit:
  ret void
}

; CHECK-LABEL: base_pointer_is_inst_inside_invariant_1
; CHECK-NOT: Valid Region for Scop

declare ptr @getNextBasePtr2(ptr) readnone nounwind

define void @base_pointer_is_inst_inside_invariant_2(i64 %n, ptr %A) {
entry:
  br label %for.i

for.i:
  %indvar.i = phi i64 [ %indvar.i.next, %for.i.inc ], [ 0, %entry ]
  %ptr = call ptr @getNextBasePtr2(ptr %A)
  %ptr2 = call ptr @getNextBasePtr(ptr %ptr)
  br label %S1

S1:
  %conv = sitofp i64 %indvar.i to float
  %arrayidx5 = getelementptr float, ptr %ptr2, i64 %indvar.i
  store float %conv, ptr %arrayidx5, align 4
  br label %for.i.inc

for.i.inc:
  %indvar.i.next = add i64 %indvar.i, 1
  %exitcond.i = icmp ne i64 %indvar.i.next, %n
  br i1 %exitcond.i, label %for.i, label %exit

exit:
  ret void
}

; CHECK-LABEL: base_pointer_is_inst_inside_invariant_2
; CHECK-NOT: Valid Region for Scop

declare ptr @getNextBasePtr3(ptr, i64) readnone nounwind

define void @base_pointer_is_inst_inside_variant(i64 %n, ptr %A) {
entry:
  br label %for.i

for.i:
  %indvar.i = phi i64 [ %indvar.i.next, %for.i.inc ], [ 0, %entry ]
  %ptr = call ptr @getNextBasePtr3(ptr %A, i64 %indvar.i)
  %ptr2 = call ptr @getNextBasePtr(ptr %ptr)
  br label %S1

S1:
  %conv = sitofp i64 %indvar.i to float
  %arrayidx5 = getelementptr float, ptr %ptr2, i64 %indvar.i
  store float %conv, ptr %arrayidx5, align 4
  br label %for.i.inc

for.i.inc:
  %indvar.i.next = add i64 %indvar.i, 1
  %exitcond.i = icmp ne i64 %indvar.i.next, %n
  br i1 %exitcond.i, label %for.i, label %exit

exit:
  ret void
}

; CHECK: base_pointer_is_inst_inside_variant
; CHECK-NOT: Valid Region for Scop

define void @base_pointer_is_ptr2ptr(ptr noalias %A, i64 %n) {
entry:
  br label %for.i

for.i:
  %indvar.i = phi i64 [ %indvar.i.next, %for.i.inc ], [ 0, %entry ]
  %arrayidx = getelementptr ptr, ptr %A, i64 %indvar.i
  br label %for.j

for.j:
  %indvar.j = phi i64 [ 0, %for.i ], [ %indvar.j.next, %for.j ]
  %conv = sitofp i64 %indvar.i to float
  %basepointer = load ptr, ptr %arrayidx, align 8
  %arrayidx5 = getelementptr float, ptr %basepointer, i64 %indvar.j
  store float %conv, ptr %arrayidx5, align 4
  %indvar.j.next = add i64 %indvar.j, 1
  %exitcond.j = icmp ne i64 %indvar.j.next, %n
  br i1 %exitcond.j, label %for.j, label %for.i.inc

for.i.inc:
  %indvar.i.next = add i64 %indvar.i, 1
  %exitcond.i = icmp ne i64 %indvar.i.next, %n
  br i1 %exitcond.i, label %for.i, label %exit

exit:
  ret void
}

; CHECK: base_pointer_is_ptr2ptr
; CHECK: Valid Region for Scop: for.j => for.i.inc
