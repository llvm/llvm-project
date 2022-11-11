; RUN: llc < %s -mtriple=x86_64-linux -asm-verbose=false | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32 -asm-verbose=false | FileCheck %s

; LSR's OptimizeMax should eliminate the select (max).

; CHECK-LABEL: foo:
; CHECK-NOT: cmov
; CHECK: jle

define void @foo(i64 %n, ptr nocapture %p) nounwind {
entry:
  %cmp6 = icmp slt i64 %n, 0                      ; <i1> [#uses=1]
  br i1 %cmp6, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %tmp = icmp sgt i64 %n, 0                       ; <i1> [#uses=1]
  %n.op = add i64 %n, 1                           ; <i64> [#uses=1]
  %tmp1 = select i1 %tmp, i64 %n.op, i64 1        ; <i64> [#uses=1]
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i = phi i64 [ %i.next, %for.body ], [ 0, %for.body.preheader ] ; <i64> [#uses=2]
  %arrayidx = getelementptr double, ptr %p, i64 %i    ; <ptr> [#uses=2]
  %t4 = load double, ptr %arrayidx                    ; <double> [#uses=1]
  %mul = fmul double %t4, 2.200000e+00            ; <double> [#uses=1]
  store double %mul, ptr %arrayidx
  %i.next = add nsw i64 %i, 1                     ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %i.next, %tmp1          ; <i1> [#uses=1]
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; In this case, one of the max operands is another max, which folds,
; leaving a two-operand max which doesn't fit the usual pattern.
; OptimizeMax should handle this case.
; PR7454

;      CHECK-LABEL: _Z18GenerateStatusPagei:

;      CHECK:         jle
;  CHECK-NOT:         cmov
;      CHECK:         xorl    {{%edi, %edi|%ecx, %ecx|%eax, %eax}}
; CHECK-NEXT:         align
; CHECK-NEXT: BB1_2:
; CHECK:              callq
; CHECK-NEXT:         incl    [[BX:%[a-z0-9]+]]
; CHECK-NEXT:         cmpl    [[R14:%[a-z0-9]+]], [[BX]]
; CHECK:              jl

define void @_Z18GenerateStatusPagei(i32 %jobs_to_display) nounwind {
entry:
  %cmp.i = icmp sgt i32 %jobs_to_display, 0       ; <i1> [#uses=1]
  %tmp = select i1 %cmp.i, i32 %jobs_to_display, i32 0 ; <i32> [#uses=3]
  %cmp8 = icmp sgt i32 %tmp, 0                    ; <i1> [#uses=1]
  br i1 %cmp8, label %bb.nph, label %for.end

bb.nph:                                           ; preds = %entry
  %tmp11 = icmp sgt i32 %tmp, 1                   ; <i1> [#uses=1]
  %smax = select i1 %tmp11, i32 %tmp, i32 1       ; <i32> [#uses=1]
  br label %for.body

for.body:                                         ; preds = %for.body, %bb.nph
  %i.010 = phi i32 [ 0, %bb.nph ], [ %inc, %for.body ] ; <i32> [#uses=1]
  %it.0.09 = phi ptr [ null, %bb.nph ], [ %call.i, %for.body ] ; <ptr> [#uses=1]
  %call.i = call ptr @_ZSt18_Rb_tree_decrementPKSt18_Rb_tree_node_base(ptr %it.0.09) ; <ptr> [#uses=1]
  %inc = add nsw i32 %i.010, 1                    ; <i32> [#uses=2]
  %exitcond = icmp eq i32 %inc, %smax             ; <i1> [#uses=1]
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare ptr @_ZSt18_Rb_tree_decrementPKSt18_Rb_tree_node_base(ptr)
