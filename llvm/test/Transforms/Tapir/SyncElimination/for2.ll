; RUN: opt < %s -sync-elimination -S | FileCheck %s
; XFAIL: *

; ModuleID = 'for2.cpp'
source_filename = "for2.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @_Z4funcv() #0 {
entry:
  %syncreg = call token @llvm.syncregion.start()
  br label %pfor.cond

pfor.cond:                                        ; preds = %pfor.inc15, %entry
  %__begin.0 = phi i32 [ 0, %entry ], [ %inc16, %pfor.inc15 ]
  %cmp = icmp slt i32 %__begin.0, 100
  br i1 %cmp, label %pfor.detach, label %pfor.cond.cleanup

pfor.cond.cleanup:                                ; preds = %pfor.cond
;; The sync before a return is not safe to remove.
; CHECK: sync within %syncreg, label %pfor.end.continue
  sync within %syncreg, label %pfor.end.continue

pfor.end.continue:                                ; preds = %pfor.cond.cleanup
  ret void

pfor.detach:                                      ; preds = %pfor.cond
  detach within %syncreg, label %pfor.body.entry, label %pfor.inc15

pfor.body.entry:                                  ; preds = %pfor.detach
  %syncreg1 = call token @llvm.syncregion.start()
  br label %pfor.body

pfor.body:                                        ; preds = %pfor.body.entry
  br label %pfor.cond5

pfor.cond5:                                       ; preds = %pfor.inc, %pfor.body
  %__begin3.0 = phi i32 [ 0, %pfor.body ], [ %inc, %pfor.inc ]
  %cmp6 = icmp slt i32 %__begin3.0, 3
  br i1 %cmp6, label %pfor.detach9, label %pfor.cond.cleanup7

; CHECK: pfor.cond5
pfor.cond.cleanup7:                               ; preds = %pfor.cond5
; CHECK-NOT: sync within %syncreg1, label %pfor.end.continue
  sync within %syncreg1, label %pfor.end.continue8
; CHECK: pfor.inc15

pfor.end.continue8:                               ; preds = %pfor.cond.cleanup7
  reattach within %syncreg, label %pfor.inc15

pfor.detach9:                                     ; preds = %pfor.cond5
  detach within %syncreg1, label %pfor.body.entry12, label %pfor.inc

pfor.body.entry12:                                ; preds = %pfor.detach9
  br label %pfor.preattach

pfor.preattach:                                   ; preds = %pfor.body.entry12
  reattach within %syncreg1, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.preattach, %pfor.detach9
  %inc = add nsw i32 %__begin3.0, 1
  br label %pfor.cond5, !llvm.loop !2

pfor.inc15:                                       ; preds = %pfor.end.continue8, %pfor.detach
  %inc16 = add nsw i32 %__begin.0, 1
  br label %pfor.cond, !llvm.loop !4
}

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }

!2 = distinct !{!2, !3}
!3 = !{!"tapir.loop.spawn.strategy", i32 1}
!4 = distinct !{!4, !3}
