; RUN: opt < %s -tapir2target -tapir-target=cilk -debug-abi-calls -S | FileCheck %s --check-prefix=TT
; RUN: opt < %s -loop-spawning -S | FileCheck %s --check-prefix=LS

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
; Function Attrs: noinline nounwind optnone uwtable
define void @detach_test() #0 {
entry:
  %x = alloca [16 x i32], align 16
  %syncreg = call token @llvm.syncregion.start()
  %arraydecay = getelementptr inbounds [16 x i32], [16 x i32]* %x, i32 0, i32 0
  call void @bar(i32* %arraydecay)
  detach within %syncreg, label %det.achd, label %det.cont

det.achd:                                         ; preds = %entry
  %y = alloca [16 x i32], align 16
  %arraydecay1 = getelementptr inbounds [16 x i32], [16 x i32]* %x, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [16 x i32], [16 x i32]* %y, i32 0, i32 0
  call void @baz(i32* %arraydecay1, i32* %arraydecay2)
  reattach within %syncreg, label %det.cont

det.cont:                                         ; preds = %det.achd, %entry
  %arraydecay3 = getelementptr inbounds [16 x i32], [16 x i32]* %x, i32 0, i32 0
  call void @bar(i32* %arraydecay3)
  sync within %syncreg, label %sync.continue

sync.continue:                                    ; preds = %det.cont
  ret void
}

; TT-LABEL: define internal fastcc void @detach_test_det.achd.cilk(
; TT: %y.cilk = alloca
; TT-LABEL: det.achd.cilk:
; TT: %[[YCILKPTRSTART:.+]] = bitcast [16 x i32]* %y.cilk to i8*
; TT-NEXT: call void @llvm.lifetime.start.p0i8(i64 64, i8* %[[YCILKPTRSTART]])
; TT: call void @baz(
; TT: %[[YCILKPTREND:.+]] = bitcast [16 x i32]* %y.cilk to i8*
; TT-NEXT: call void @llvm.lifetime.end.p0i8(i64 64, i8* %[[YCILKPTREND]])

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

declare void @bar(i32*) local_unnamed_addr #2

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

declare void @baz(i32*, i32*) local_unnamed_addr #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1
; Function Attrs: nounwind uwtable
define void @ploop_test(i32 %n) local_unnamed_addr #0 {
entry:
  %x = alloca [16 x i32], align 16
  %syncreg = tail call token @llvm.syncregion.start()
  %syncreg6 = tail call token @llvm.syncregion.start()
  %0 = bitcast [16 x i32]* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %0) #3
  %arraydecay = getelementptr inbounds [16 x i32], [16 x i32]* %x, i64 0, i64 0
  call void @bar(i32* nonnull %arraydecay) #3
  %cmp47 = icmp sgt i32 %n, 0
  br i1 %cmp47, label %pfor.detach.preheader, label %pfor.cond.cleanup

pfor.detach.preheader:                            ; preds = %entry
  br label %pfor.detach

pfor.cond.cleanup:                                ; preds = %pfor.inc, %entry
  sync within %syncreg, label %sync.continue

pfor.detach:                                      ; preds = %pfor.detach.preheader, %pfor.inc
  %__begin.048 = phi i32 [ %inc, %pfor.inc ], [ 0, %pfor.detach.preheader ]
  detach within %syncreg, label %pfor.body, label %pfor.inc

pfor.body:                                        ; preds = %pfor.detach
  %y = alloca [16 x i32], align 16
  %1 = bitcast [16 x i32]* %y to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %1) #3
  %arraydecay4 = getelementptr inbounds [16 x i32], [16 x i32]* %y, i64 0, i64 0
  call void @baz(i32* nonnull %arraydecay, i32* nonnull %arraydecay4) #3
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %1) #3
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.body, %pfor.detach
  %inc = add nuw nsw i32 %__begin.048, 1
  %exitcond49 = icmp eq i32 %inc, %n
  br i1 %exitcond49, label %pfor.cond.cleanup, label %pfor.detach, !llvm.loop !2

sync.continue:                                    ; preds = %pfor.cond.cleanup
  %cmp1545 = icmp sgt i32 %n, 0
  br i1 %cmp1545, label %pfor.detach17.preheader, label %pfor.cond.cleanup16

pfor.detach17.preheader:                          ; preds = %sync.continue
  br label %pfor.detach17

pfor.cond.cleanup16:                              ; preds = %pfor.inc26, %sync.continue
  sync within %syncreg6, label %sync.continue28

pfor.detach17:                                    ; preds = %pfor.detach17.preheader, %pfor.inc26
  %__begin8.046 = phi i32 [ %inc27, %pfor.inc26 ], [ 0, %pfor.detach17.preheader ]
  detach within %syncreg6, label %pfor.body22, label %pfor.inc26

pfor.body22:                                      ; preds = %pfor.detach17
  %2 = alloca [16 x i32], align 16
  %.sub = getelementptr inbounds [16 x i32], [16 x i32]* %2, i64 0, i64 0
  call void @baz(i32* nonnull %arraydecay, i32* nonnull %.sub) #3
  reattach within %syncreg6, label %pfor.inc26

pfor.inc26:                                       ; preds = %pfor.body22, %pfor.detach17
  %inc27 = add nuw nsw i32 %__begin8.046, 1
  %exitcond = icmp eq i32 %inc27, %n
  br i1 %exitcond, label %pfor.cond.cleanup16, label %pfor.detach17, !llvm.loop !4

sync.continue28:                                  ; preds = %pfor.cond.cleanup16
  call void @bar(i32* nonnull %arraydecay) #3
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %0) #3
  ret void
}

; LS-LABEL: define internal fastcc void @ploop_test_pfor.detach.ls(
; LS: %y.ls = alloca
; LS: pfor.body.ls:
; LS: %[[YLSPTRSTART:.+]] = bitcast [16 x i32]* %y.ls to i8*
; LS-NEXT: call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %[[YLSPTRSTART]])
; LS: call void @baz(
; LS-NEXT: call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %[[YLSPTRSTART]])

; LS-LABEL: define internal fastcc void @ploop_test_pfor.detach17.ls(
; LS: %[[ALLOCACALLPTR:.+]] = alloca
; LS: pfor.body22.ls:
; LS: %[[ALLOCACALLPTRSTART:.+]] = bitcast [16 x i32]* %[[ALLOCACALLPTR]] to i8*
; LS-NEXT: call void @llvm.lifetime.start.p0i8(i64 64, i8* %[[ALLOCACALLPTRSTART]])
; LS: call void @baz(
; LS: %[[ALLOCACALLPTREND:.+]] = bitcast [16 x i32]* %[[ALLOCACALLPTR]] to i8*
; LS-NEXT: call void @llvm.lifetime.end.p0i8(i64 64, i8* %[[ALLOCACALLPTREND]])

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!2 = distinct !{!2, !3}
!3 = !{!"tapir.loop.spawn.strategy", i32 1}
!4 = distinct !{!4, !3}
