; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; BasicAA should detect NoAliases in PHIs and Selects.

; CHECK: Function: foo
; CHECK:  NoAlias: double* %a, double* %b
; CHECK: Function: bar
; CHECK:  NoAlias: double* %a, double* %b
; CHECK: Function: qux
; CHECK:  NoAlias: double* %a, double* %b
; CHECK: Function: fin
; CHECK:  NoAlias: double* %a, double* %b
; CHECK: ===== Alias Analysis Evaluator Report =====

; Two PHIs in the same block.
define void @foo(i1 %m, double* noalias %x, double* noalias %y) {
entry:
  br i1 %m, label %true, label %false

true:
  br label %exit

false:
  br label %exit

exit:
  %a = phi double* [ %x, %true ], [ %y, %false ]
  %b = phi double* [ %x, %false ], [ %y, %true ]
  store volatile double 0.0, double* %a
  store volatile double 1.0, double* %b
  ret void
}

; Two selects with the same condition.
define void @bar(i1 %m, double* noalias %x, double* noalias %y) {
entry:
  %a = select i1 %m, double* %x, double* %y
  %b = select i1 %m, double* %y, double* %x
  store volatile double 0.000000e+00, double* %a
  store volatile double 1.000000e+00, double* %b
  ret void
}

; Two PHIs with disjoint sets of inputs.
define void @qux(i1 %m, double* noalias %x, double* noalias %y,
                 i1 %n, double* noalias %v, double* noalias %w) {
entry:
  br i1 %m, label %true, label %false

true:
  br label %exit

false:
  br label %exit

exit:
  %a = phi double* [ %x, %true ], [ %y, %false ]
  br i1 %n, label %ntrue, label %nfalse

ntrue:
  br label %nexit

nfalse:
  br label %nexit

nexit:
  %b = phi double* [ %v, %ntrue ], [ %w, %nfalse ]
  store volatile double 0.0, double* %a
  store volatile double 1.0, double* %b
  ret void
}

; Two selects with disjoint sets of arms.
define void @fin(i1 %m, double* noalias %x, double* noalias %y,
                 i1 %n, double* noalias %v, double* noalias %w) {
entry:
  %a = select i1 %m, double* %x, double* %y
  %b = select i1 %n, double* %v, double* %w
  store volatile double 0.000000e+00, double* %a
  store volatile double 1.000000e+00, double* %b
  ret void
}
