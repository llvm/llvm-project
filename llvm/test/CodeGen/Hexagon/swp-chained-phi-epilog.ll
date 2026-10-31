; RUN: llc -mtriple=hexagon -mcpu=hexagonv81 -O2 < %s | FileCheck %s
; RUN: llc -mtriple=hexagon -mcpu=hexagonv81 -O2 -debug-only=pipeliner \
; RUN:   < %s 2>&1 | FileCheck %s --check-prefix=EXPANDER
; REQUIRES: asserts

; A two-channel biquad IIR recurrence carries four pieces of state (x1, x2,
; y1, y2) across the loop. The state forms chained loop-carried PHIs: y2 is a
; PHI whose loop value is the PHI y1, scheduled in an earlier stage. The
; default modulo-schedule expander cannot rotate this recurrence through the
; epilog and emitted a stale y2 on the final iteration. Such
; loops are expanded with the peeling expander instead, which is correct, and
; the loop stays software pipelined.

; EXPANDER: Using peeling expander for chained loop-carried PHIs

; The EXPANDER check above is the regression guard: without the fix the loop is
; still pipelined (the loop0/endloop0 checks below pass pre-patch), so only the
; expander selection distinguishes correct from stale-epilog codegen.

; CHECK-LABEL: fused_2ch:
; CHECK: loop0(
; CHECK: endloop0

declare float @llvm.fmuladd.f32(float, float, float)

define void @fused_2ch(float %b0, float %b1, float %b2, float %a1, float %a2,
                       ptr noalias nocapture readonly %in,
                       ptr noalias nocapture writeonly %out, i32 %ns) #0 {
entry:
  %nn = shl nsw i32 %ns, 1
  %pos = icmp sgt i32 %nn, 0
  br i1 %pos, label %for.body, label %exit

for.body:
  %x1a = phi float [ 0.0, %entry ], [ %xa, %for.body ]
  %x2a = phi float [ 0.0, %entry ], [ %x1a, %for.body ]
  %y1a = phi float [ 0.0, %entry ], [ %ya, %for.body ]
  %y2a = phi float [ 0.0, %entry ], [ %y1a, %for.body ]
  %x1b = phi float [ 0.0, %entry ], [ %xb, %for.body ]
  %x2b = phi float [ 0.0, %entry ], [ %x1b, %for.body ]
  %y1b = phi float [ 0.0, %entry ], [ %yb, %for.body ]
  %y2b = phi float [ 0.0, %entry ], [ %y1b, %for.body ]
  %inp = phi ptr [ %in, %entry ], [ %inn, %for.body ]
  %outp = phi ptr [ %out, %entry ], [ %outn, %for.body ]
  %n = phi i32 [ 0, %entry ], [ %nnext, %for.body ]
  %xa = load float, ptr %inp, align 4
  %idx = or disjoint i32 %n, 1
  %inp1 = getelementptr inbounds float, ptr %in, i32 %idx
  %xb = load float, ptr %inp1, align 4
  %ma = fmul float %b0, %xa
  %a2v = tail call float @llvm.fmuladd.f32(float %b1, float %x1a, float %ma)
  %a3v = tail call float @llvm.fmuladd.f32(float %b2, float %x2a, float %a2v)
  %a4v = tail call float @llvm.fmuladd.f32(float %a1, float %y1a, float %a3v)
  %ya  = tail call float @llvm.fmuladd.f32(float %a2, float %y2a, float %a4v)
  %mb = fmul float %b0, %xb
  %b6v = tail call float @llvm.fmuladd.f32(float %b1, float %x1b, float %mb)
  %b7v = tail call float @llvm.fmuladd.f32(float %b2, float %x2b, float %b6v)
  %b8v = tail call float @llvm.fmuladd.f32(float %a1, float %y1b, float %b7v)
  %yb  = tail call float @llvm.fmuladd.f32(float %a2, float %y2b, float %b8v)
  store float %ya, ptr %outp, align 4
  %outp1 = getelementptr inbounds float, ptr %out, i32 %idx
  store float %yb, ptr %outp1, align 4
  %nnext = add nuw nsw i32 %n, 2
  %inn = getelementptr i8, ptr %inp, i32 8
  %outn = getelementptr i8, ptr %outp, i32 8
  %cmp = icmp slt i32 %nnext, %nn
  br i1 %cmp, label %for.body, label %exit

exit:
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv81" }
