; RUN: opt %loadNPMPolly -passes=polly-codegen -S < %s | FileCheck %s
;
; llvm.org/PR25439
; Scalar reloads in the generated entering block were not recognized as
; dominating the subregion blocks when there were multiple entering nodes. This
; resulted in values defined in there (here: %cond used in subregionB_entry) not
; being copied. We check whether it is reusing the reloaded scalar.
;
; CHECK-LABEL: polly.stmt.subregionB_entry.exit:
; CHECK:         store i1 %polly.cond, ptr %cond.s2a
;
; CHECK-LABEL: polly.stmt.subregionB_entry.entry:
; CHECK:         %cond.s2a.reload = load i1, ptr %cond.s2a
;
; CHECK-LABEL: polly.stmt.subregionB_entry:
; CHECK:         br i1 %cond.s2a.reload

define void @func(ptr %A) {
entry:
  br label %subregionA_entry

subregionA_entry:
  %cond = phi i1 [ false, %entry ], [ true, %subregionB_exit ]
  br i1 %cond, label %subregionA_if, label %subregionA_else

subregionA_if:
  br label %subregionB_entry

subregionA_else:
  br label %subregionB_entry

subregionB_entry:
  store i32 0, ptr %A
  br i1 %cond, label %subregionB_if, label %subregionB_exit

subregionB_if:
  br label %subregionB_exit

subregionB_exit:
  br i1 false, label %subregionA_entry, label %return

return:
  ret void
}
