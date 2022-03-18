; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-process-unprofitable=false -polly-unprofitable-scalar-accs=false -polly-prune-unprofitable -disable-output -stats < %s 2>&1 | FileCheck -match-full-lines %s
; RUN: opt %loadNPMPolly -polly-stmt-granularity=bb -polly-process-unprofitable=false -polly-unprofitable-scalar-accs=false "-passes=scop(polly-prune-unprofitable)" -disable-output -stats < %s 2>&1 | FileCheck -match-full-lines %s
; REQUIRES: asserts
;
; Skip this SCoP for having scalar dependencies between all statements,
; but only after ScopInfo (because optimization passes using ScopInfo such
; as DeLICM might remove these scalar dependencies).
;
; double x = 0;
; for (int i = 0; i < n; i += 1)
;   for (int j = 0; j < m; j += 1) {
;      B[0] = x;
;      x = A[0];
;   }
; return x;
;
define double @func(i32 %n, i32 %m, double* noalias nonnull %A, double* noalias nonnull %B) {
entry:
  br label %outer.for

outer.for:
  %outer.phi = phi double [0.0, %entry], [%inner.phi, %outer.inc]
  %i = phi i32 [0, %entry], [%i.inc, %outer.inc]
  %i.cmp = icmp slt i32 %i, %n
  br i1 %i.cmp, label %inner.for, label %outer.exit

    inner.for:
      %inner.phi = phi double [%outer.phi, %outer.for], [%load, %inner.inc]
      %j = phi i32 [0, %outer.for], [%j.inc, %inner.inc]
      %j.cmp = icmp slt i32 %j, %m
      br i1 %j.cmp, label %body, label %inner.exit

        body:
          store double %inner.phi, double* %B
          %load = load double, double* %A
          br label %inner.inc

    inner.inc:
      %j.inc = add nuw nsw i32 %j, 1
      br label %inner.for

    inner.exit:
      br label %outer.inc

outer.inc:
  %i.inc = add nuw nsw i32 %i, 1
  br label %outer.for

outer.exit:
  br label %return

return:
  ret double %outer.phi
}


; CHECK: 1 polly-prune-unprofitable - Number of pruned SCoPs because it they cannot be optimized in a significant way
