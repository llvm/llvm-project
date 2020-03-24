; RUN: opt < %s -licm -S | FileCheck %s

; Function Attrs: noinline nounwind uwtable
define void @normalize(double* noalias %out, double* noalias %in, i32 %n) #0 {
; CHECK-LABEL: @normalize(
entry:
  %syncreg = call token @llvm.syncregion.start()
  %cmp1 = icmp slt i32 0, %n
  br i1 %cmp1, label %pfor.detach.lr.ph, label %pfor.end

pfor.detach.lr.ph:                                ; preds = %entry
; CHECK: pfor.detach.lr.ph:
; CHECK-NEXT: %call = call double @norm(double* %in, i32 %n)
  br label %pfor.detach

pfor.detach:                                      ; preds = %pfor.detach.lr.ph, %pfor.inc
  %i.02 = phi i32 [ 0, %pfor.detach.lr.ph ], [ %inc, %pfor.inc ]
  detach within %syncreg, label %pfor.body, label %pfor.inc

pfor.body:                                        ; preds = %pfor.detach
; CHECK-NOT: call double @norm(
  %idxprom = sext i32 %i.02 to i64
  %arrayidx = getelementptr inbounds double, double* %in, i64 %idxprom
  %0 = load double, double* %arrayidx, align 8
  ;; Should have hoisted this call
  %call = call double @norm(double* %in, i32 %n) #2
  %div = fdiv double %0, %call
  %idxprom1 = sext i32 %i.02 to i64
  %arrayidx2 = getelementptr inbounds double, double* %out, i64 %idxprom1
  store double %div, double* %arrayidx2, align 8
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.body, %pfor.detach
  %inc = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %pfor.detach, label %pfor.cond.pfor.end_crit_edge, !llvm.loop !1

pfor.cond.pfor.end_crit_edge:                     ; preds = %pfor.inc
  br label %pfor.end

pfor.end:                                         ; preds = %pfor.cond.pfor.end_crit_edge, %entry
  sync within %syncreg, label %pfor.end.continue

pfor.end.continue:                                ; preds = %pfor.end
  ret void
}

; Function Attrs: nounwind readonly
declare double @norm(double*, i32) #1

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #3

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { argmemonly nounwind }

!1 = distinct !{!1, !2}
!2 = !{!"tapir.loop.spawn.strategy", i32 1}
