; RUN: opt %loadNPMPolly -passes=polly-codegen -S < %s | FileCheck %s
;
; CHECK: polly.merge_new_and_old:
; CHECK:   %result.ph.merge = phi float [ %result.ph.final_reload, %polly.exiting ], [ %result.ph, %next.region_exiting ]
; CHECK:   br label %next
;
; CHECK: next:
; CHECK:   %result = phi float [ %result.ph.merge, %polly.merge_new_and_old ]
; CHECK:   ret float %result

define float @foo(ptr %A, i64 %param) {
entry:
  br label %entry.split

entry.split:
  %branchcond = icmp slt i64 %param, 64
  br i1 %branchcond, label %loopA, label %loopB

loopA:
  %indvarA = phi i64 [0, %entry.split], [%indvar.nextA, %loopA]
  %indvar.nextA = add i64 %indvarA, 1
  %valA = load float, ptr %A
  %sumA = fadd float %valA, %valA
  store float %valA, ptr %A
  %cndA = icmp eq i64 %indvar.nextA, 100
  br i1 %cndA, label %next, label %loopA

loopB:
  %indvarB = phi i64 [0, %entry.split], [%indvar.nextB, %loopB]
  %indvar.nextB = add i64 %indvarB, 1
  %valB = load float, ptr %A
  %sumB = fadd float %valB, %valB
  store float %valB, ptr %A
  %cndB = icmp eq i64 %indvar.nextB, 100
  br i1 %cndB, label %next, label %loopB

next:
  %result = phi float [%sumA, %loopA], [%sumB, %loopB]
  ret float %result

}
