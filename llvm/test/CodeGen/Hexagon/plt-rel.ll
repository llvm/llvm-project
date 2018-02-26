; RUN: llc -march=hexagon -relocation-model=pic -mattr=+long-calls < %s | FileCheck --check-prefix=CHECK-LONG %s
; RUN: llc -march=hexagon -relocation-model=pic < %s | FileCheck %s

; CHECK-LONG: call ##_ZL13g_usr1_called@GDPLT
; CHECK-LONG-NOT: call _ZL13g_usr1_called@GDPLT
; CHECK: call _ZL13g_usr1_called@GDPLT
; CHECK-NOT: call ##_ZL13g_usr1_called@GDPLT


target triple = "hexagon"

@_ZL13g_usr1_called = internal thread_local global i32 0, align 4

; Function Attrs: norecurse nounwind
define void @_Z14SigUsr1Handleri(i32) local_unnamed_addr #0 {
entry:
  store volatile i32 1, i32* @_ZL13g_usr1_called, align 4
  ret void
}

; Function Attrs: norecurse nounwind
define zeroext i1 @_Z27CheckForMonitorCancellationv() local_unnamed_addr #0 {
entry:
  %0 = load volatile i32, i32* @_ZL13g_usr1_called, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %return, label %if.then

if.then:                                          ; preds = %entry
  store volatile i32 0, i32* @_ZL13g_usr1_called, align 4
  br label %return

return:                                           ; preds = %entry, %if.then
  %.sink = phi i1 [ true, %if.then ], [ false, %entry ]
  ret i1 %.sink
}

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
