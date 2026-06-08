; RUN: opt -passes=loop-load-elim -S < %s | FileCheck %s

; LoopLoadElimination could call LoopVersioning on non-LCSSA loops.
; A direct exit-block use of %C then remained tied to the original loop,
; which stopped dominating the shared exit after cloning. Forming LCSSA in
; versionLoop() rewrites the branch through an exit PHI; Usage of raw %C and %C.exit is needed
; to trigger the old buggy path.

define void @non_lcssa_exit_use(ptr nocapture %a, i64 %n) {
; CHECK-LABEL: @non_lcssa_exit_use(
; CHECK:       for.body.lver.check:
; CHECK:       for.end:
; CHECK:         %C.lcssa = phi i1
; CHECK:         br i1 %C.lcssa,
; CHECK-NOT:     br i1 %C,
entry:
  %G = getelementptr i32, ptr %a, i64 -1
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %G, i64 %iv
  %load = load i32, ptr %arrayidx, align 4
  %mul = mul i32 %load, 3
  %arrayidx2 = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %mul, ptr %arrayidx2, align 4
  %iv.next = add i64 %iv, 1
  %C = icmp sgt i64 %iv.next, %n
  br i1 %C, label %for.end, label %for.body

for.end:
  %C.exit = phi i1 [ %C, %for.body ]
  br i1 %C, label %exit.a, label %exit.b

exit.a:
  ret void

exit.b:
  ret void
}
