; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s
;
; llvm.org/PR25438
; After loop versioning, a dominance check of a non-affine subregion's exit node
; causes the dominance check to always fail any block in the scop. The
; subregion's exit block has become polly_merge_new_and_old, which also receives
; the control flow of the generated code. This would cause that any value for
; implicit stores is assumed to be not from the scop.
;
; This checks that the stored value is indeed from the generated code.
;
; CHECK-LABEL: polly.stmt.do.body.entry:
; CHECK:        a.phiops.reload = load i32, i32* %a.phiops
;
; CHECK-LABEL: polly.stmt.polly.merge_new_and_old.exit:
; CHECK:         store i32 %polly.a, i32* %a.s2a

define void @func() {
entry:
  br label %while.body

while.body:
  br label %do.body

do.body:
  %a = phi i32 [ undef, %while.body ], [ %b, %end_b ]
  %cond = or i1 undef, undef
  br i1 %cond, label %end_a, label %if_a

if_a:
  br label %end_a

end_a:
  br i1 undef, label %if_b, label %end_b

if_b:
  br label %end_b

end_b:
  %b = phi i32 [ undef, %if_b ], [ %a, %end_a ]
  br i1 false, label %do.body, label %do.end

do.end:
  br label %return

return:
  ret void
}
