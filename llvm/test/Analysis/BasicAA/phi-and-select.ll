; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; BasicAA should detect NoAliases in PHIs and Selects.

; Two PHIs in the same block.
; CHECK-LABEL: Function: foo
; CHECK: NoAlias: double* %a, double* %b
define void @foo(i1 %m, ptr noalias %x, ptr noalias %y) {
entry:
  br i1 %m, label %true, label %false

true:
  br label %exit

false:
  br label %exit

exit:
  %a = phi ptr [ %x, %true ], [ %y, %false ]
  %b = phi ptr [ %x, %false ], [ %y, %true ]
  store volatile double 0.0, ptr %a
  store volatile double 1.0, ptr %b
  ret void
}

; Two selects with the same condition.
; CHECK-LABEL: Function: bar
; CHECK: NoAlias: double* %a, double* %b
define void @bar(i1 %m, ptr noalias %x, ptr noalias %y) {
entry:
  %a = select i1 %m, ptr %x, ptr %y
  %b = select i1 %m, ptr %y, ptr %x
  store volatile double 0.000000e+00, ptr %a
  store volatile double 1.000000e+00, ptr %b
  ret void
}

; Two PHIs with disjoint sets of inputs.
; CHECK-LABEL: Function: qux
; CHECK: NoAlias: double* %a, double* %b
define void @qux(i1 %m, ptr noalias %x, ptr noalias %y,
                 i1 %n, ptr noalias %v, ptr noalias %w) {
entry:
  br i1 %m, label %true, label %false

true:
  br label %exit

false:
  br label %exit

exit:
  %a = phi ptr [ %x, %true ], [ %y, %false ]
  br i1 %n, label %ntrue, label %nfalse

ntrue:
  br label %nexit

nfalse:
  br label %nexit

nexit:
  %b = phi ptr [ %v, %ntrue ], [ %w, %nfalse ]
  store volatile double 0.0, ptr %a
  store volatile double 1.0, ptr %b
  ret void
}

; Two selects with disjoint sets of arms.
; CHECK-LABEL: Function: fin
; CHECK: NoAlias: double* %a, double* %b
define void @fin(i1 %m, ptr noalias %x, ptr noalias %y,
                 i1 %n, ptr noalias %v, ptr noalias %w) {
entry:
  %a = select i1 %m, ptr %x, ptr %y
  %b = select i1 %n, ptr %v, ptr %w
  store volatile double 0.000000e+00, ptr %a
  store volatile double 1.000000e+00, ptr %b
  ret void
}

; A gep off a select of two argument pointers does not alias a noalias argument.
; CHECK-LABEL: Function: select_and_gep_unknown_size
; CHECK: NoModRef:   call void @llvm.memset.p0.i32(ptr %g, i8 0, i32 %size, i1 false) <->   call void @llvm.memset.p0.i32(ptr %z, i8 0, i32 %size, i1 false)
define void @select_and_gep_unknown_size(i1 %c, ptr %x, ptr %y, ptr noalias %z, i32 %size) {
entry:
  %p = select i1 %c, ptr %x, ptr %y
  %g = getelementptr inbounds i8, ptr %p, i64 1
  call void @llvm.memset.p0.i32(ptr %g, i8 0, i32 %size, i1 false)
  call void @llvm.memset.p0.i32(ptr %z, i8 0, i32 %size, i1 false)
  ret void
}

; If a select arm is the other pointer, the gep may alias it.
; CHECK-LABEL: Function: select_and_gep_unknown_size_may
; CHECK: MayAlias:	i8* %g, i8* %z
define void @select_and_gep_unknown_size_may(i1 %c, ptr %x, ptr %z) {
entry:
  %p = select i1 %c, ptr %x, ptr %z
  %g = getelementptr inbounds i8, ptr %p, i64 1
  store i8 0, ptr %g
  store i8 0, ptr %z
  ret void
}

declare void @llvm.memset.p0.i32(ptr, i8, i32, i1)

; On the first iteration, sel1 = a1, sel2 = a2, phi = a3
; On the second iteration, sel1 = a2, sel1 = a1, phi = a2
; As such, sel1 and phi may alias.
; CHECK-LABEL: Function: select_backedge
; CHECK: NoAlias:	i32* %sel1, i32* %sel2
; CHECK: MayAlias:	i32* %phi, i32* %sel1
; CHECK: MayAlias:	i32* %phi, i32* %sel2
define void @select_backedge() {
entry:
  %a1 = alloca i32
  %a2 = alloca i32
  %a3 = alloca i32
  br label %loop

loop:
  %phi = phi ptr [ %a3, %entry ], [ %sel2, %loop ]
  %c = phi i1 [ true, %entry ], [ false, %loop ]
  %sel1 = select i1 %c, ptr %a1, ptr %a2
  %sel2 = select i1 %c, ptr %a2, ptr %a1
  load i32, ptr %sel1
  load i32, ptr %sel2
  load i32, ptr %phi
  br label %loop
}
