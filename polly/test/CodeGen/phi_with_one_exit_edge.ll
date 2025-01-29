; RUN: opt %loadNPMPolly -passes=polly-codegen -S < %s | FileCheck %s
;
;
; CHECK: polly.merge_new_and_old:
; CHECK:   %sumA.merge = phi float [ %sumA.final_reload, %polly.exiting ], [ %sumA, %loopA ]
; CHECK:   br label %next
;
; CHECK: next:
; CHECK:   %result = phi float [ %sumA.merge, %polly.merge_new_and_old ]
; CHECK:   ret float %result
;
define float @foo(ptr %A, i64 %param) {
entry:
  br label %entry.split

entry.split:
  br label %loopA

loopA:
  %indvarA = phi i64 [0, %entry.split], [%indvar.nextA, %loopA]
  %indvar.nextA = add i64 %indvarA, 1
  %valA = load float, ptr %A
  %sumA = fadd float %valA, %valA
  store float %valA, ptr %A
  %cndA = icmp eq i64 %indvar.nextA, 100
  br i1 %cndA, label %next, label %loopA

next:
  %result = phi float [%sumA, %loopA]
  ret float %result

}
