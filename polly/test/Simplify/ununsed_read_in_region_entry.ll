; RUN: opt %loadPolly -polly-print-simplify -disable-output< %s | FileCheck %s -match-full-lines
; RUN: opt %loadPolly -polly-simplify -polly-codegen -S < %s | FileCheck %s -check-prefix=CODEGEN
;
; for (int i = 0; i < n; i+=1) {
;    (void)A[0];
;    if (21.0 == 0.0)
;      B[0] = 42.0;
; }
;
define void @func(i32 %n, double* noalias nonnull %A, double* noalias nonnull %B) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %region_entry, label %exit


    region_entry:
      %val = load double, double* %A
      %cmp = fcmp oeq double 21.0, 0.0
      br i1 %cmp, label %region_true, label %region_exit

    region_true:
      store double 42.0, double* %B
      br label %region_exit

    region_exit:
      br label %body

    body:
      br label %inc


inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}


; CHECK: Statistics {
; CHECK:     Dead accesses removed: 1
; CHECK:     Dead instructions removed: 1
; CHECK: }

; CHECK:      After accesses {
; CHECK-NEXT:     Stmt_region_entry__TO__region_exit
; CHECK-NEXT:             MayWriteAccess :=   [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_region_entry__TO__region_exit[i0] -> MemRef_B[0] };
; CHECK-NEXT: }


; CODEGEN:      polly.stmt.region_entry:
; CODEGEN-NEXT:   %p_cmp = fcmp oeq double 2.100000e+01, 0.000000e+00
; CODEGEN-NEXT:   br i1 %p_cmp
