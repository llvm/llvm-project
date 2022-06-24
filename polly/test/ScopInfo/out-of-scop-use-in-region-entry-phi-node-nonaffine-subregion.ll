; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s
;
; Check whether %newval is identified as escaping value, even though it is used
; in a phi that is in the region. Non-affine subregion case.
;
; CHECK-LABEL: subregion_entry.region_entering:
; CHECK:         %loop_carried.ph = phi float [ %newval.merge, %backedge ], [ undef, %entry ]
;
; CHECK-LABEL: polly.merge_new_and_old:
; CHECK:         %newval.merge = phi float [ %newval.final_reload, %polly.exiting ], [ %newval, %subregion_exit.region_exiting ]
;
; CHECK-LABEL: polly.start:
; CHECK:         store float %loop_carried.ph, ptr %loop_carried.phiops
;
; CHECK-LABEL: polly.stmt.subregion_entry.entry:
; CHECK:         %loop_carried.phiops.reload = load float, ptr %loop_carried.phiops
;
; CHECK-LABEL: polly.stmt.subregion_entry:
; CHECK:         %polly.loop_carried = phi float [ %loop_carried.phiops.reload, %polly.stmt.subregion_entry.entry ]
; CHECK:         %p_newval = fadd float %polly.loop_carried, 1.000000e+00
;
; CHECK-LABEL: polly.stmt.polly.merge_new_and_old.exit:
; CHECK:         %newval.final_reload = load float, ptr %newval.s2a

define void @func() {
entry:
  br label %subregion_entry

subregion_entry:
  %loop_carried = phi float [ undef, %entry ], [ %newval, %backedge ]
  %indvar = phi i32 [ 1, %entry ], [ %indvar_next, %backedge ]
  %newval = fadd float %loop_carried, 1.0
  %cmp = fcmp ogt float undef, undef
  br i1 %cmp, label %subregion_if, label %subregion_exit

subregion_if:
  br label %subregion_exit

subregion_exit:
  br i1 undef, label %if_then, label %if_else

if_then:
  br label %backedge

if_else:
  br label %backedge

backedge:
  %indvar_next = add nuw nsw i32 %indvar, 1
  br i1 false, label %subregion_entry, label %exit

exit:
  ret void
}
