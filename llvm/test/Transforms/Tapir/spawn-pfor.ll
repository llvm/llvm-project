; RUN: opt < %s -task-simplify -S -o - 2>&1 | FileCheck %s
; RUN: opt < %s -passes="task-simplify" -S -o - 2>&1 | FileCheck %s

; ModuleID = 'spawn-pfor.c'
source_filename = "spawn-pfor.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @foo(i32 %n) local_unnamed_addr #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %syncreg3 = tail call token @llvm.syncregion.start()
  detach within %syncreg, label %det.achd, label %det.cont

det.achd:                                         ; preds = %entry
  tail call void @bar(i32 0) #3
  reattach within %syncreg, label %det.cont

det.cont:                                         ; preds = %det.achd, %entry
  detach within %syncreg, label %det.achd1, label %det.cont2

det.achd1:                                        ; preds = %det.cont
  tail call void @bar(i32 1) #3
  reattach within %syncreg, label %det.cont2

det.cont2:                                        ; preds = %det.achd1, %det.cont
  %cmp19 = icmp sgt i32 %n, 0
  br i1 %cmp19, label %pfor.detach.preheader, label %pfor.cond.cleanup

pfor.detach.preheader:                            ; preds = %det.cont2
  br label %pfor.detach

pfor.cond.cleanup:                                ; preds = %pfor.inc, %det.cont2
  sync within %syncreg3, label %sync.continue

pfor.detach:                                      ; preds = %pfor.detach.preheader, %pfor.inc
  %__begin.020 = phi i32 [ %inc, %pfor.inc ], [ 0, %pfor.detach.preheader ]
  detach within %syncreg3, label %pfor.body, label %pfor.inc

pfor.body:                                        ; preds = %pfor.detach
  %add6 = add nuw nsw i32 %__begin.020, 2
  tail call void @bar(i32 %add6) #3
  reattach within %syncreg3, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.body, %pfor.detach
  %inc = add nuw nsw i32 %__begin.020, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %pfor.cond.cleanup, label %pfor.detach, !llvm.loop !2

sync.continue:                                    ; preds = %pfor.cond.cleanup
  tail call void @bar(i32 %n) #3
  sync within %syncreg, label %sync.continue10

sync.continue10:                                  ; preds = %sync.continue
  ret void
}

; CHECK: entry:
; CHECK-NEXT: %syncreg = tail call token @llvm.syncregion.start
; CHECK-NEXT: %syncreg3 = tail call token @llvm.syncregion.start

; CHECK: pfor.cond.cleanup:
; CHECK-NEXT: sync within %syncreg3,

; CHECK: sync.continue:
; CHECK-NEXT: tail call void @bar
; CHECK-NEXT: sync within %syncreg,

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

declare void @bar(i32) local_unnamed_addr #2

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.0 (git@github.com:wsmoses/Tapir-Clang.git 96fbe7006d96197be05a8c45720a2b1d281e1678) (git@github.com:wsmoses/Tapir-LLVM.git 8a0ce31c7dd131c39642b9097a00fe3bcc18bb81)"}
!2 = distinct !{!2, !3}
!3 = !{!"tapir.loop.spawn.strategy", i32 1}
