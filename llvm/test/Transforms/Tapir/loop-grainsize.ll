; RUN: opt < %s -loop-spawning-ti -S | FileCheck %s
; RUN: opt < %s -passes='loop-spawning' -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: uwtable
define i32 @_Z12pragma_testsi(i32 %n) local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp23 = icmp sgt i32 %n, 0
  br i1 %cmp23, label %pfor.detach.preheader, label %pfor.cond.cleanup

pfor.detach.preheader:                            ; preds = %entry
  br label %pfor.detach
; CHECK: {{^pfor.detach.preheader}}
; CHECK-NEXT: invoke fastcc void @_Z12pragma_testsi.outline_pfor.detach.ls1(i32 0, i32 %n, i32 4)

pfor.cond.cleanup:                                ; preds = %pfor.inc, %entry
  sync within %syncreg, label %sync.continue

pfor.detach:                                      ; preds = %pfor.detach.preheader, %pfor.inc
  %__begin.024 = phi i32 [ %inc, %pfor.inc ], [ 0, %pfor.detach.preheader ]
  detach within %syncreg, label %pfor.body, label %pfor.inc unwind label %lpad3.loopexit

pfor.body:                                        ; preds = %pfor.detach
  %call = invoke i32 @_Z3fooi(i32 %__begin.024)
          to label %pfor.preattach unwind label %lpad

pfor.preattach:                                   ; preds = %pfor.body
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.detach, %pfor.preattach
  %inc = add nuw nsw i32 %__begin.024, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %pfor.cond.cleanup, label %pfor.detach, !llvm.loop !2

lpad:                                             ; preds = %pfor.body
  %0 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg, { i8*, i32 } %0)
          to label %det.rethrow.unreachable unwind label %lpad3.loopexit.split-lp

det.rethrow.unreachable:                          ; preds = %lpad
  unreachable

lpad3.loopexit:                                   ; preds = %pfor.detach
  %lpad.loopexit = landingpad { i8*, i32 }
          cleanup
  br label %lpad3

lpad3.loopexit.split-lp:                          ; preds = %lpad
  %lpad.loopexit.split-lp = landingpad { i8*, i32 }
          cleanup
  br label %lpad3

lpad3:                                            ; preds = %lpad3.loopexit.split-lp, %lpad3.loopexit
  %lpad.phi = phi { i8*, i32 } [ %lpad.loopexit, %lpad3.loopexit ], [ %lpad.loopexit.split-lp, %lpad3.loopexit.split-lp ]
  sync within %syncreg, label %sync.continue7

sync.continue:                                    ; preds = %pfor.cond.cleanup
  ret i32 0

sync.continue7:                                   ; preds = %lpad3
  resume { i8*, i32 } %lpad.phi
}

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

declare i32 @_Z3fooi(i32) local_unnamed_addr #2

declare i32 @__gxx_personality_v0(...)

; Function Attrs: argmemonly
declare void @llvm.detached.rethrow.sl_p0i8i32s(token, { i8*, i32 }) #3

attributes #0 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { argmemonly }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.0 (git@github.com:wsmoses/Tapir-Clang.git 4243d6a74e292ae62b82f7ff71233f8a2aeb4481) (git@github.mit.edu:SuperTech/Tapir-CSI-llvm.git 23d12922c9b8bcbec235e208eb6b60a2dcee6451)"}
!2 = distinct !{!2, !3, !4}
!3 = !{!"tapir.loop.spawn.strategy", i32 1}
!4 = !{!"tapir.loop.grainsize", i32 4}
