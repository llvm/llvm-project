; Verify that task-simplify preserves both sync regions in this
; example code.
;
; RUN: opt < %s -task-simplify -S -o - | FileCheck %s
; RUN: opt < %s -passes='task-simplify' -S -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@globl = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: norecurse nounwind uwtable
define dso_local void @bar() local_unnamed_addr #0 {
entry:
  %0 = load i32, i32* @globl, align 4, !tbaa !2
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @globl, align 4, !tbaa !2
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @foo(i32 %n) local_unnamed_addr #1 {
entry:
; CHECK-LABEL: entry:
  %sum = alloca i32, align 4
  %syncreg = tail call token @llvm.syncregion.start()
; CHECK: %syncreg = tail call token @llvm.syncregion
  %syncreg1 = tail call token @llvm.syncregion.start()
; CHECK: %syncreg1 = tail call token @llvm.syncregion
  %sum.0.sum.0..sroa_cast = bitcast i32* %sum to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %sum.0.sum.0..sroa_cast)
  store i32 0, i32* %sum, align 4, !tbaa !2
  detach within %syncreg, label %det.achd, label %det.cont
; CHECK: detach within %syncreg, label %det.achd

det.achd:                                         ; preds = %entry
  tail call void @bar()
  reattach within %syncreg, label %det.cont

det.cont:                                         ; preds = %det.achd, %entry
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %pfor.cond, label %cleanup

pfor.cond:                                        ; preds = %det.cont, %pfor.inc
  %__begin.0 = phi i32 [ %inc, %pfor.inc ], [ 0, %det.cont ]
  detach within %syncreg1, label %pfor.body, label %pfor.inc
; CHECK: pfor.cond:
; CHECK: detach within %syncreg1, label %pfor.body

pfor.body:                                        ; preds = %pfor.cond
  %sum.0.load21 = load i32, i32* %sum, align 4
  %add4 = add nsw i32 %sum.0.load21, %__begin.0
  store i32 %add4, i32* %sum, align 4, !tbaa !2
  reattach within %syncreg1, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.body, %pfor.cond
  %inc = add nuw nsw i32 %__begin.0, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %pfor.cond.cleanup, label %pfor.cond, !llvm.loop !6

pfor.cond.cleanup:                                ; preds = %pfor.inc
  sync within %syncreg1, label %cleanup
; CHECK: pfor.cond.cleanup:
; CHECK: sync within %syncreg1

cleanup:                                          ; preds = %pfor.cond.cleanup, %det.cont
  tail call void @bar()
  sync within %syncreg, label %sync.continue7
; CHECK: cleanup:
; CHECK: sync within %syncreg

sync.continue7:                                   ; preds = %cleanup
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %sum.0.sum.0..sroa_cast)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #2

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #2

attributes #0 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { argmemonly nounwind }
attributes #3 = { inlinehint nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind readonly }
attributes #6 = { nounwind }

!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = distinct !{!6, !7}
!7 = !{!"tapir.loop.spawn.strategy", i32 1}
!8 = !{!9, !9, i64 0}
!9 = !{!"any pointer", !4, i64 0}
