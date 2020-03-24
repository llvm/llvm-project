; RUN: opt < %s -loop-spawning-ti -S 2>&1 | FileCheck %s
; RUN: opt < %s -passes=loop-spawning -S 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define noalias double* @_Z17new_array_no_initmb(i64 %n, i1 zeroext %touch_pages) local_unnamed_addr #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %0 = lshr i64 %n, 3
  %add = shl i64 %0, 6
  %mul1 = add i64 %add, 64
  %call = tail call noalias i8* @aligned_alloc(i64 64, i64 %mul1) #3
  %1 = bitcast i8* %call to double*
  br i1 %touch_pages, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %cmp15 = icmp eq i64 %mul1, 0
  br i1 %cmp15, label %pfor.cond.cleanup, label %pfor.detach.preheader

pfor.detach.preheader:                            ; preds = %if.then
  br label %pfor.detach

pfor.cond.cleanup:                                ; preds = %pfor.inc, %if.then
  sync within %syncreg, label %if.end

pfor.detach:                                      ; preds = %pfor.detach.preheader, %pfor.inc
  %i.016 = phi i64 [ %add2, %pfor.inc ], [ 0, %pfor.detach.preheader ]
  detach within %syncreg, label %pfor.body, label %pfor.inc

pfor.body:                                        ; preds = %pfor.detach
  %arrayidx = getelementptr inbounds i8, i8* %call, i64 %i.016
  store i8 0, i8* %arrayidx, align 1, !tbaa !2
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.body, %pfor.detach
  %add2 = add i64 %i.016, 2097152
  %cmp = icmp ult i64 %add2, %mul1
  br i1 %cmp, label %pfor.detach, label %pfor.cond.cleanup, !llvm.loop !6

if.end:                                           ; preds = %pfor.cond.cleanup, %entry
  ret double* %1
}

; CHECK: Tapir loop not transformed: failed to use divide-and-conquer loop spawning

; Function Attrs: nounwind
declare noalias i8* @aligned_alloc(i64, i64) local_unnamed_addr #1

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #2

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { argmemonly nounwind }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.0 (git@github.com:wsmoses/Tapir-Clang.git 131c6308b501a74f21a086c456381fa10810f7f7) (git@github.mit.edu:SuperTech/Tapir-CSI-llvm.git f885b3a2f5a6c58740774c9bdbd32cc025500ad2)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"bool", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = distinct !{!6, !7}
!7 = !{!"tapir.loop.spawn.strategy", i32 1}
