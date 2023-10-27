; RUN: llc < %s | FileCheck %s

target triple = "aarch64"


define void @dont_rematerialize_cntd(i32 %N) #0 {
; CHECK-LABEL: dont_rematerialize_cntd:
; CHECK:        cntd
; CHECK:        smstop sm
; CHECK-NOT:    cntd
; CHECK:        bl      foo
; CHECK:        smstart  sm
entry:
  %cmp2 = icmp sgt i32 %N, 0
  br i1 %cmp2, label %for.body, label %for.cond.cleanup

for.body:                                         ; preds = %entry, %for.body
  %index.03 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  call void asm sideeffect "", "~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27}"() nounwind
  %.tr = call i32 @llvm.vscale.i32()
  %conv = shl nuw nsw i32 %.tr, 4
  call void @foo(i32 %conv)
  %inc = add nuw nsw i32 %index.03, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void
}

; This test doesn't strictly make sense, because it passes a scalable predicate
; to a function, which makes little sense if the VL is not the same in/out of
; streaming-SVE mode. If the VL is known to be the same, then we could just as
; well rematerialize the `ptrue` inside the call sequence. However, the purpose
; of this test is more to ensure that the logic works, which may also trigger
; when the value is not being passed as argument (e.g. when it is hoisted from
; a loop and placed inside the call sequence).
;
; FIXME: This test also exposes another bug, where the 'mul vl' addressing mode
; is used before/after the smstop. This will be fixed in a future patch.
define void @dont_rematerialize_ptrue(i32 %N) #0 {
; CHECK-LABEL: dont_rematerialize_ptrue:
; CHECK:        ptrue [[PTRUE:p[0-9]+]].b
; CHECK:        str [[PTRUE]], [[[SPILL_ADDR:.*]]]
; CHECK:        smstop sm
; CHECK:        ldr p0, [[[SPILL_ADDR]]]
; CHECK-NOT:    ptrue
; CHECK:        bl      bar
; CHECK:        smstart  sm
entry:
  %cmp2 = icmp sgt i32 %N, 0
  br i1 %cmp2, label %for.body, label %for.cond.cleanup

for.body:                                         ; preds = %entry, %for.body
  %index.03 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  call void asm sideeffect "", "~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27}"() nounwind
  %ptrue.ins = insertelement <vscale x 16 x i1> poison, i1 1, i32 0
  %ptrue = shufflevector <vscale x 16 x i1> %ptrue.ins, <vscale x 16 x i1> poison, <vscale x 16 x i32> zeroinitializer
  call void @bar(<vscale x 16 x i1> %ptrue)
  %inc = add nuw nsw i32 %index.03, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void
}
declare void @foo(i32)
declare void @bar(<vscale x 16 x i1>)
declare i32 @llvm.vscale.i32()

attributes #0 = { "aarch64_pstate_sm_enabled" "frame-pointer"="non-leaf" "target-features"="+sme,+sve" }
