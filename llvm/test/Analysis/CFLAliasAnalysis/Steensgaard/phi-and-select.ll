; RUN: opt < %s -aa-pipeline=cfl-steens-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
; Derived from (a subset of) BasicAA/phi-and-select.ll 

; CHECK: Function: qux
; CHECK:  NoAlias: double* %a, double* %b
; CHECK: ===== Alias Analysis Evaluator Report =====

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

