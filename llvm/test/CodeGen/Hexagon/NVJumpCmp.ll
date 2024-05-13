; RUN: llc -march=hexagon -O2 -mcpu=hexagonv60  < %s | FileCheck %s

; Look for an instruction, we really just do not want to see an abort.
; CHECK: trace_event
; REQUIRES: asserts

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a:0-n16:32"
target triple = "hexagon-unknown--elf"

; Function Attrs: nounwind
define void @_ZN6Halide7Runtime8Internal13default_traceEPvPK18halide_trace_event() #0 {
entry:
  br i1 undef, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %if.then
  br i1 undef, label %while.cond, label %while.end

while.end:                                        ; preds = %while.cond
  %add = add i32 undef, 48
  br i1 undef, label %if.end, label %if.then17

if.then17:                                        ; preds = %while.end
  unreachable

if.end:                                           ; preds = %while.end
  %arrayidx21 = getelementptr inbounds [4096 x i8], ptr undef, i32 0, i32 8
  store i8 undef, ptr %arrayidx21, align 4, !tbaa !1
  br i1 undef, label %for.body42.preheader6, label %min.iters.checked

for.body42.preheader6:                            ; preds = %vector.body.preheader, %min.iters.checked, %if.end
  unreachable

min.iters.checked:                                ; preds = %if.end
  br i1 undef, label %for.body42.preheader6, label %vector.body.preheader

vector.body.preheader:                            ; preds = %min.iters.checked
  br i1 undef, label %for.cond48.preheader, label %for.body42.preheader6

for.cond48.preheader:                             ; preds = %vector.body.preheader
  br i1 undef, label %while.cond.i, label %for.body61.lr.ph

for.body61.lr.ph:                                 ; preds = %for.cond48.preheader
  br i1 undef, label %for.body61, label %min.iters.checked595

min.iters.checked595:                             ; preds = %for.body61.lr.ph
  br i1 undef, label %for.body61, label %vector.memcheck608

vector.memcheck608:                               ; preds = %min.iters.checked595
  %scevgep600 = getelementptr [4096 x i8], ptr undef, i32 0, i32 %add
  %bound0604 = icmp ule ptr %scevgep600, undef
  %memcheck.conflict607 = and i1 undef, %bound0604
  br i1 %memcheck.conflict607, label %for.body61, label %vector.body590

vector.body590:                                   ; preds = %vector.body590, %vector.memcheck608
  br i1 undef, label %middle.block591, label %vector.body590, !llvm.loop !4

middle.block591:                                  ; preds = %vector.body590
  %cmp.n613 = icmp eq i32 undef, 0
  br i1 %cmp.n613, label %while.cond.i, label %for.body61

while.cond.i:                                     ; preds = %for.body61, %while.cond.i, %middle.block591, %for.cond48.preheader
  br i1 undef, label %_ZN6Halide7Runtime8Internal14ScopedSpinLockC2EPVi.exit, label %while.cond.i

_ZN6Halide7Runtime8Internal14ScopedSpinLockC2EPVi.exit: ; preds = %while.cond.i
  unreachable

for.body61:                                       ; preds = %for.body61, %middle.block591, %vector.memcheck608, %min.iters.checked595, %for.body61.lr.ph
  %cmp59 = icmp ult i32 undef, undef
  br i1 %cmp59, label %for.body61, label %while.cond.i, !llvm.loop !7

if.else:                                          ; preds = %entry
  unreachable
}

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"halide_mattrs", !"+hvx"}
!1 = !{!2, !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = distinct !{!4, !5, !6}
!5 = !{!"llvm.loop.vectorize.width", i32 1}
!6 = !{!"llvm.loop.interleave.count", i32 1}
!7 = distinct !{!7, !5, !6}
