; assert in DAGlegalizer with fake use of 1-element vectors.
; RUN: llc -stop-after=finalize-isel -mtriple=x86_64-unknown-linux -filetype=asm -o - %s | FileCheck %s
;
; ModuleID = 't2.cpp'
; source_filename = "t2.cpp"
; target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
;
; Check that we get past ISel and generate FAKE_USE machine instructions for
; one-element vectors.
;
; CHECK:       bb.0.entry:
; CHECK-DAG:   %1:gr64 = COPY $rdi
; CHECK-DAG:   %0:vr128 = COPY $xmm0
; CHECK:       %2:vr64 =
; CHECK-DAG:   FAKE_USE %1
; CHECK-DAG:   FAKE_USE %0
; CHECK:       RET


target triple = "x86_64-unknown-unknown"

; Function Attrs: nounwind sspstrong uwtable
define <4 x float> @_Z3runDv4_fDv1_x(<4 x float> %r, i64 %b.coerce) local_unnamed_addr #0 {
entry:
  %0 = insertelement <1 x i64> undef, i64 %b.coerce, i32 0
  %1 = bitcast i64 %b.coerce to <1 x i64>
  %2 = tail call <4 x float> @llvm.x86.sse.cvtpi2ps(<4 x float> %r, <1 x i64> %1)
  tail call void (...) @llvm.fake.use(<1 x i64> %0)
  tail call void (...) @llvm.fake.use(<4 x float> %r)
  ret <4 x float> %2
}

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.x86.sse.cvtpi2ps(<4 x float>, <1 x i64>)

; Function Attrs: nounwind
declare void @llvm.fake.use(...)

attributes #0 = { "target-cpu"="btver2" optdebug }
