; RUN: opt < %s -loop-stripmine -S -o - | FileCheck %s
; RUN: opt < %s -passes='loop-stripmine' -S -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: argmemonly nounwind uwtable
define dso_local void @daxpy(double* nocapture %y, double* nocapture readonly %x, double %a, i32 %n) local_unnamed_addr #0 !dbg !6 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp15 = icmp sgt i32 %n, 0, !dbg !8
  br i1 %cmp15, label %pfor.detach.preheader, label %pfor.cond.cleanup, !dbg !8

pfor.detach.preheader:                            ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %pfor.detach, !dbg !8

pfor.cond.cleanup:                                ; preds = %pfor.inc, %entry
  sync within %syncreg, label %sync.continue, !dbg !8

pfor.detach:                                      ; preds = %pfor.inc, %pfor.detach.preheader
  %indvars.iv = phi i64 [ 0, %pfor.detach.preheader ], [ %indvars.iv.next, %pfor.inc ]
  detach within %syncreg, label %pfor.body, label %pfor.inc, !dbg !8

pfor.body:                                        ; preds = %pfor.detach
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv, !dbg !9
  %0 = load double, double* %arrayidx, align 8, !dbg !9, !tbaa !10
  %mul3 = fmul double %0, %a, !dbg !14
  %arrayidx5 = getelementptr inbounds double, double* %y, i64 %indvars.iv, !dbg !15
  %1 = load double, double* %arrayidx5, align 8, !dbg !16, !tbaa !10
  %add6 = fadd double %1, %mul3, !dbg !16
  store double %add6, double* %arrayidx5, align 8, !dbg !16, !tbaa !10
  reattach within %syncreg, label %pfor.inc, !dbg !15

pfor.inc:                                         ; preds = %pfor.body, %pfor.detach
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !8
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count, !dbg !8
  br i1 %exitcond, label %pfor.cond.cleanup, label %pfor.detach, !dbg !8, !llvm.loop !17

sync.continue:                                    ; preds = %pfor.cond.cleanup
  ret void, !dbg !20
}

; Test for the basic structure of the stripmined loop with an epilog.

; CHECK-LABEL: define dso_local void @daxpy(
; CHECK: pfor.detach.preheader:
; CHECK:  %[[STRPLOOPITER:.+]] = add i64 %[[TRIPCOUNT:.+]], -1
; CHECK:  %[[XTRAITER:.+]] = and i64 %[[TRIPCOUNT]], 2047
; CHECK:  %[[ICMP:.+]] = icmp ult i64 %[[STRPLOOPITER]], 2047
; CHECK:  br i1 %[[ICMP]], label %[[EPILCHECK:.+]], label %[[STRPLOOPPH:.+]], !dbg !8

; CHECK: [[STRPLOOPPH]]:
; CHECK-NEXT: %[[STRPITER:.+]] = udiv i64 %[[TRIPCOUNT]], 2048
; CHECK-NEXT: br label %[[STRPLOOPOUTERHEAD:.+]], !dbg !8

; CHECK: reattach within %syncreg, label %[[STRPLOOPOUTERINC:.+]], !dbg !8

; CHECK: [[STRPLOOPOUTERINC]]:
; CHECK-NEXT: %[[STRPLOOPITERNEXT:.+]] = add nuw nsw i64 %[[STRPLOOPITER:.+]], 1
; CHECK-NEXT: %[[STRPLOOPLATCHCMP:.+]] = icmp eq i64 %[[STRPLOOPITERNEXT]], %[[STRPITER]]
; CHECK-NEXT: br i1 %[[STRPLOOPLATCHCMP]], label %[[STRPLOOPOUTEREXIT:.+]], label %[[STRPLOOPOUTERHEAD]]

; CHECK: [[STRPLOOPOUTEREXIT]]:
; CHECK: br label %[[EPILCHECK]]

; CHECK: [[EPILCHECK]]:
; CHECK-NEXT: %[[ICMPEPIL:.+]] = icmp ne i64 %[[XTRAITER]], 0
; CHECK-NEXT: br i1 %[[ICMPEPIL]], label %[[EPILPH:.+]], label %[[LOOPEXIT:.+]], !dbg !8

; CHECK: [[EPILPH]]:
; CHECK-NEXT: %[[EPILSTART:.+]] = sub nuw nsw i64 %[[TRIPCOUNT]], %[[XTRAITER]]
; CHECK-NEXT: br label %[[EPILHEAD:.+]], !dbg !8

; CHECK: [[EPILHEAD]]:
; CHECK-NEXT: %[[EPILIV:.+]] = phi i64
; CHECK-DAG: [ %[[EPILSTART]], %[[EPILPH]] ]
; CHECK-DAG: [ %[[EPILNEXT:.+]], %[[EPILINC:.+]] ]
; CHECK-NEXT: %[[EPILITER:.+]] = phi i64
; CHECK-DAG: [ %[[XTRAITER]], %[[EPILPH]] ]
; CHECK-DAG: [ %[[ITERNEXT:.+]], %[[EPILINC]] ]
; CHECK-NEXT: br label %[[EPILBODY:.+]], !dbg !8

; CHECK: [[EPILBODY]]:
; CHECK: getelementptr {{.+}}i64 %[[EPILIV]]
; CHECK: getelementptr {{.+}}i64 %[[EPILIV]]
; CHECK: br label %[[EPILINC]]

; CHECK: [[EPILINC]]:
; CHECK-DAG: %[[EPILNEXT]] = add nuw nsw i64 %[[EPILIV]]
; CHECK-DAG: %[[ITERNEXT]] = sub nsw i64 %[[EPILITER]]
; CHECK-NEXT: %[[EPILLATCHCMP:.+]] = icmp ne i64 %[[ITERNEXT]], 0
; CHECK-NEXT: br i1 %[[EPILLATCHCMP]], label %[[EPILHEAD]], label

; CHECK: pfor.cond.cleanup:
; CHECK-NEXT: sync within %syncreg

; CHECK: [[STRPLOOPOUTERHEAD]]:
; CHECK-NEXT: %[[STRPLOOPITER]] = phi i64
; CHECK-DAG: [ 0, %[[STRPLOOPPH]] ]
; CHECK-DAG: [ %[[STRPLOOPITERNEXT]], %[[STRPLOOPOUTERINC]] ]
; CHECK-NEXT: detach within %syncreg, label %[[STRPLOOPTASKENTRY:.+]], label %[[STRPLOOPOUTERINC]], !dbg !8

; CHECK: [[STRPLOOPTASKENTRY]]:
; CHECK-NEXT: %[[STRPLOOPSTART:.+]] = mul i64 2048, %[[STRPLOOPITER]], !dbg !8
; CHECK-NEXT: br label %[[STRPLOOPINNERHEAD:.+]], !dbg !8

; CHECK: [[STRPLOOPINNERHEAD]]:
; CHECK-NEXT: %[[STRPLOOPIV:.+]] = phi i64
; CHECK: [ %[[STRPLOOPSTART]], %[[STRPLOOPTASKENTRY]] ]
; CHECK-NEXT: %[[STRPLOOPINNERITER:.+]] = phi i64
; CHECK: [ 2048, %[[STRPLOOPTASKENTRY]] ]
; CHECK-NEXT: br label %[[STRPLOOPINNERBODY:.+]], !dbg !8

; CHECK: [[STRPLOOPINNERBODY]]:
; CHECK: getelementptr {{.+}}i64 %[[STRPLOOPIV]]
; CHECK: getelementptr {{.+}}i64 %[[STRPLOOPIV]]
; CHECK: br label %[[STRPLOOPINNERINC:.+]], !dbg

; CHECK: [[STRPLOOPINNERINC]]:
; CHECK-DAG: %[[STRPLOOPNEXT:.+]] = add nuw nsw i64 %[[STRPLOOPIV]], 1
; CHECK-DAG: %[[STRPLOOPINNERNEXT:.+]] = sub nuw nsw i64 %[[STRPLOOPINNERITER]], 1
; CHECK: %[[STRPLOOPINNERLATCHCMP:.+]] = icmp eq i64 %[[STRPLOOPINNERNEXT]], 0
; CHECK: br i1 %[[STRPLOOPINNERLATCHCMP]], label {{.+}}, label %[[STRPLOOPINNERHEAD]]

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind uwtable
define dso_local void @daxpy2(double* nocapture %y, double* nocapture readonly %x, double %a, i32 %n) local_unnamed_addr #0 !dbg !21 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp15 = icmp sgt i32 %n, 0, !dbg !22
  br i1 %cmp15, label %pfor.detach.preheader, label %pfor.cond.cleanup, !dbg !22

pfor.detach.preheader:                            ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %pfor.detach, !dbg !22

pfor.cond.cleanup:                                ; preds = %pfor.inc, %entry
  sync within %syncreg, label %sync.continue, !dbg !22

pfor.detach:                                      ; preds = %pfor.inc, %pfor.detach.preheader
  %indvars.iv = phi i64 [ 0, %pfor.detach.preheader ], [ %indvars.iv.next, %pfor.inc ]
  detach within %syncreg, label %pfor.body, label %pfor.inc, !dbg !22

pfor.body:                                        ; preds = %pfor.detach
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv, !dbg !23
  %0 = load double, double* %arrayidx, align 8, !dbg !23, !tbaa !10
  %mul3 = fmul double %0, %a, !dbg !24
  %arrayidx5 = getelementptr inbounds double, double* %y, i64 %indvars.iv, !dbg !25
  %1 = load double, double* %arrayidx5, align 8, !dbg !26, !tbaa !10
  %add6 = fadd double %1, %mul3, !dbg !26
  store double %add6, double* %arrayidx5, align 8, !dbg !26, !tbaa !10
  reattach within %syncreg, label %pfor.inc, !dbg !25

pfor.inc:                                         ; preds = %pfor.body, %pfor.detach
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !22
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count, !dbg !22
  br i1 %exitcond, label %pfor.cond.cleanup, label %pfor.detach, !dbg !22, !llvm.loop !27

sync.continue:                                    ; preds = %pfor.cond.cleanup
  ret void, !dbg !30
}

; Test for detaching the stripmined loop from the epilog.

; CHECK-LABEL: define dso_local void @daxpy2(
; CHECK: pfor.detach.preheader:
; CHECK:  %[[STRPLOOPITER:.+]] = add i64 %[[TRIPCOUNT:.+]], -1
; CHECK:  %[[XTRAITER:.+]] = and i64 %[[TRIPCOUNT]], 15
; CHECK:  %[[ICMP:.+]] = icmp ult i64 %[[STRPLOOPITER]], 15
; CHECK:  br i1 %[[ICMP]], label %[[EPILCHECK:.+]], label %[[STRPLOOPPH:.+]], !dbg !26

; CHECK: [[STRPLOOPPH]]:
; CHECK-NEXT: br label %[[STRPLOOPDETACH:.+]], !dbg !26

; CHECK: [[STRPLOOPDETACH]]:
; CHECK-NEXT: detach within %syncreg, label %[[STRPLOOPDETACHENTRY:.+]], label %[[STRPLOOPDETACHCONT:.+]], !dbg !26

; CHECK: [[STRPLOOPDETACHCONT]]:
; CHECK-NEXT: br label %[[EPILCHECK]]

; CHECK: [[STRPLOOPDETACHENTRY]]:
; CHECK-NEXT: %[[NESTEDSYNCREG:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: udiv i64 %[[TRIPCOUNT]], 16
; CHECK-NEXT: br label %[[STRPLOOPOUTERHEAD:.+]], !dbg !26

; CHECK: [[STRPLOOPOUTERHEAD]]:
; CHECK-NEXT: %[[STRPLOOPOUTERITER:.+]] = phi i64
; CHECK-NEXT: detach within %[[NESTEDSYNCREG]], label %[[STRPLOOPOUTERENTRY:.+]], label %[[STRPLOOPOUTERINC:.+]], !dbg !26

; CHECK: [[STRPLOOPOUTERENTRY]]:
; CHECK-NEXT: mul i64 16, %[[STRPLOOPOUTERITER]], !dbg !26
; CHECK-NEXT: br label %[[STRPLOOPINNERHEAD:.+]], !dbg !26

; CHECK: [[STRPLOOPINNERHEAD]]:
; CHECK: br label %[[STRPLOOPINNERBODY:.+]], !dbg !26

; CHECK: [[STRPLOOPINNERBODY]]:
; CHECK: br label %[[STRPLOOPINNERINC:.+]], !dbg !29

; CHECK: [[STRPLOOPINNERINC]]:
; CHECK: br i1 {{.+}}, label %[[STRPLOOPINNEREXIT:.+]], label %[[STRPLOOPINNERHEAD]], !dbg !26

; CHECK: [[STRPLOOPINNEREXIT]]:
; CHECK-NEXT: reattach within %[[NESTEDSYNCREG]], label %[[STRPLOOPOUTERINC]]

; CHECK: [[STRPLOOPOUTERINC]]:
; CHECK: br i1 {{.+}}, label %[[STRPLOOPSYNC:.+]], label %[[STRPLOOPOUTERHEAD]]

; CHECK: [[STRPLOOPSYNC]]:
; CHECK-NEXT: sync within %[[NESTEDSYNCREG]], label %[[STRPLOOPREATTACH:.+]], !dbg !26

; CHECK: [[STRPLOOPREATTACH]]:
; CHECK-NEXT: reattach within %syncreg, label %[[STRPLOOPDETACHCONT]]

; Function Attrs: nounwind uwtable
define dso_local void @dsiny(double* nocapture %y, double* nocapture readonly %x, i32 %n) local_unnamed_addr #2 !dbg !31 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp27 = icmp sgt i32 %n, 0, !dbg !32
  br i1 %cmp27, label %pfor.detach, label %pfor.cond.cleanup, !dbg !32

pfor.cond.cleanup:                                ; preds = %pfor.inc, %entry
  sync within %syncreg, label %sync.continue, !dbg !32

pfor.detach:                                      ; preds = %entry, %pfor.inc
  %indvars.iv = phi i32 [ %indvars.iv.next, %pfor.inc ], [ 0, %entry ]
  %__begin.028 = phi i32 [ %inc12, %pfor.inc ], [ 0, %entry ]
  detach within %syncreg, label %pfor.body, label %pfor.inc, !dbg !32

pfor.body:                                        ; preds = %pfor.detach
  %mul3 = shl i32 %__begin.028, 12, !dbg !33
  %mul5 = add i32 %mul3, 4096, !dbg !34
  %cmp625 = icmp eq i32 %mul3, 2147479552, !dbg !35
  br i1 %cmp625, label %for.cond.cleanup, label %for.body.preheader, !dbg !36

for.body.preheader:                               ; preds = %pfor.body
  %0 = zext i32 %indvars.iv to i64, !dbg !32
  br label %for.body, !dbg !37

for.cond.cleanup:                                 ; preds = %for.body, %pfor.body
  reattach within %syncreg, label %pfor.inc, !dbg !38

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv29 = phi i64 [ %indvars.iv.next30, %for.body ], [ %0, %for.body.preheader ]
  %j.026 = phi i32 [ %inc, %for.body ], [ %mul3, %for.body.preheader ]
  %sub9 = sub nsw i32 4095, %j.026, !dbg !37
  %idxprom = sext i32 %sub9 to i64, !dbg !39
  %arrayidx = getelementptr inbounds double, double* %x, i64 %idxprom, !dbg !39
  %1 = load double, double* %arrayidx, align 8, !dbg !39, !tbaa !10
  %call = tail call double @sin(double %1) #4, !dbg !40
  %arrayidx11 = getelementptr inbounds double, double* %y, i64 %indvars.iv29, !dbg !41
  store double %call, double* %arrayidx11, align 8, !dbg !42, !tbaa !10
  %indvars.iv.next30 = add nuw nsw i64 %indvars.iv29, 1, !dbg !43
  %inc = add nuw nsw i32 %j.026, 1, !dbg !43
  %2 = trunc i64 %indvars.iv.next30 to i32, !dbg !35
  %cmp6 = icmp sgt i32 %mul5, %2, !dbg !35
  br i1 %cmp6, label %for.body, label %for.cond.cleanup, !dbg !36, !llvm.loop !44

pfor.inc:                                         ; preds = %for.cond.cleanup, %pfor.detach
  %inc12 = add nuw nsw i32 %__begin.028, 1, !dbg !32
  %indvars.iv.next = add i32 %indvars.iv, 4096, !dbg !32
  %exitcond = icmp eq i32 %inc12, %n, !dbg !32
  br i1 %exitcond, label %pfor.cond.cleanup, label %pfor.detach, !dbg !32, !llvm.loop !45

sync.continue:                                    ; preds = %pfor.cond.cleanup
  ret void, !dbg !46
}

; Function Attrs: nounwind
declare dso_local double @sin(double) local_unnamed_addr #3

; Test that large parallel loops are not stripmined.

; CHECK-LABEL: define dso_local void @dsiny(
; CHECK: pfor.detach.preheader:
; CHECK-NOT: br i1 %{{.+}}, label %{{.+}}, label %{{.+}}, !dbg
; CHECK: br label %pfor.detach

; CHECK: pfor.detach:
; CHECK-NEXT: %indvars.iv
; CHECK-NEXT: %__begin.028
; CHECK-NOT: br label %pfor.body
; CHECK-NEXT: detach within %syncreg, label %pfor.body

; Function Attrs: nounwind uwtable
define dso_local void @bar(i32 %n) local_unnamed_addr #2 !dbg !47 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp12 = icmp sgt i32 %n, 0, !dbg !48
  br i1 %cmp12, label %pfor.detach.preheader, label %pfor.cond.cleanup, !dbg !48

pfor.detach.preheader:                            ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %pfor.detach, !dbg !48

pfor.cond.cleanup:                                ; preds = %pfor.inc, %entry
  sync within %syncreg, label %sync.continue, !dbg !48

pfor.detach:                                      ; preds = %pfor.inc, %pfor.detach.preheader
  %indvars.iv = phi i64 [ 0, %pfor.detach.preheader ], [ %indvars.iv.next, %pfor.inc ]
  detach within %syncreg, label %pfor.body, label %pfor.inc, !dbg !48

pfor.body:                                        ; preds = %pfor.detach
  %B = alloca [7 x double], align 16
  %0 = call i8* @llvm.stacksave(), !dbg !49
  %vla = alloca double, i64 %indvars.iv, align 16, !dbg !49
  %1 = bitcast [7 x double]* %B to i8*, !dbg !49
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %1) #4, !dbg !49
  %2 = trunc i64 %indvars.iv to i32, !dbg !50
  %call = call i32 @foo(double* nonnull %vla, i32 %2) #4, !dbg !50
  %arraydecay = getelementptr inbounds [7 x double], [7 x double]* %B, i64 0, i64 0, !dbg !51
  %call3 = call i32 @foo(double* nonnull %arraydecay, i32 7) #4, !dbg !52
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %1) #4, !dbg !53
  call void @llvm.stackrestore(i8* %0), !dbg !53
  reattach within %syncreg, label %pfor.inc, !dbg !53

pfor.inc:                                         ; preds = %pfor.body, %pfor.detach
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !48
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count, !dbg !48
  br i1 %exitcond, label %pfor.cond.cleanup, label %pfor.detach, !dbg !48, !llvm.loop !54

sync.continue:                                    ; preds = %pfor.cond.cleanup
  ret void, !dbg !56
}

; Function Attrs: nounwind
declare i8* @llvm.stacksave() #4

; Function Attrs: nounwind
declare void @llvm.stackrestore(i8*) #4

declare dso_local i32 @foo(double*, i32) local_unnamed_addr #5

; Test the placement of stack allocations in the stripmined loop and epilog.

; CHECK-LABEL: define dso_local void @bar(
; CHECK: alloca [7 x double]

; CHECK: pfor.detach.preheader:
; CHECK:  %[[STRPLOOPITER:.+]] = add i64 %[[TRIPCOUNT:.+]], -1
; CHECK:  %[[XTRAITER:.+]] = and i64 %[[TRIPCOUNT]], 31
; CHECK:  %[[ICMP:.+]] = icmp ult i64 %[[STRPLOOPITER]], 31
; CHECK:  br i1 %[[ICMP]], label %[[EPILCHECK:.+]], label %[[STRPLOOPPH:.+]], !dbg !53

; CHECK: [[STRPLOOPPH]]:
; CHECK-NEXT: br label %[[STRPLOOPDETACH:.+]], !dbg !53

; CHECK: [[STRPLOOPDETACH]]:
; CHECK-NEXT: detach within %syncreg, label %[[STRPLOOPDETACHENTRY:.+]], label %[[STRPLOOPDETACHCONT:.+]], !dbg !53

; CHECK: [[STRPLOOPDETACHCONT]]:
; CHECK-NEXT: br label %[[EPILCHECK]]

; CHECK: [[EPILCHECK]]:
; CHECK: br i1 {{.+}}, label %[[EPILPH:.+]], label

; CHECK: [[EPILPH]]:
; CHECK: br label %[[EPILHEADER:.+]], !dbg !53

; CHECK: [[EPILHEADER]]:
; CHECK: br label %[[EPILBODY:.+]], !dbg !53

; CHECK: [[EPILBODY]]:
; CHECK-NEXT: call i8* @llvm.stacksave()
; CHECK: alloca double
; CHECK: call void @llvm.lifetime.start
; CHECK: call void @llvm.lifetime.end
; CHECK: call void @llvm.stackrestore(

; CHECK: [[STRPLOOPDETACHENTRY]]:
; CHECK: %[[NEWSYNCREG:.+]] = call token @llvm.syncregion.start()
; CHECK: br label %[[STRPLOOPOUTERHEAD:.+]], !dbg !53

; CHECK: [[STRPLOOPOUTERHEAD]]:
; CHECK: detach within %[[NEWSYNCREG]], label %[[STRPLOOPINNERENTRY:.+]], label %{{.+}}, !dbg !53

; CHECK: [[STRPLOOPINNERENTRY]]:
; CHECK: alloca [7 x double]
; CHECK: br label %[[STRPLOOPINNERHEAD:.+]], !dbg !53

; CHECK: [[STRPLOOPINNERHEAD]]:
; CHECK: br label %[[STRPLOOPINNERBODY:.+]], !dbg !53

; CHECK: [[STRPLOOPINNERBODY]]:
; CHECK: call i8* @llvm.stacksave()
; CHECK: alloca double
; CHECK: call void @llvm.lifetime.start
; CHECK: call void @llvm.lifetime.end
; CHECK: call void @llvm.stackrestore(

attributes #0 = { argmemonly nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }
attributes #5 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 7.0.0 (git@github.com:wsmoses/Tapir-Clang.git dc9a8d2e98c088903c7ad63b576d7e73de94f4a1) (git@github.com:wsmoses/Tapir-LLVM.git 3c50c14938a10d8d50b534caa34022c4dbbc4569)", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "daxpy.c", directory: "/data/compilers/tests/adhoc/tapirloops")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 7.0.0 (git@github.com:wsmoses/Tapir-Clang.git dc9a8d2e98c088903c7ad63b576d7e73de94f4a1) (git@github.com:wsmoses/Tapir-LLVM.git 3c50c14938a10d8d50b534caa34022c4dbbc4569)"}
!6 = distinct !DISubprogram(name: "daxpy", scope: !1, file: !1, line: 5, type: !7, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 6, column: 3, scope: !6)
!9 = !DILocation(line: 7, column: 17, scope: !6)
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C/C++ TBAA"}
!14 = !DILocation(line: 7, column: 15, scope: !6)
!15 = !DILocation(line: 7, column: 5, scope: !6)
!16 = !DILocation(line: 7, column: 10, scope: !6)
!17 = distinct !{!17, !8, !18, !19}
!18 = !DILocation(line: 7, column: 20, scope: !6)
!19 = !{!"tapir.loop.spawn.strategy", i32 1}
!20 = !DILocation(line: 8, column: 1, scope: !6)
!21 = distinct !DISubprogram(name: "daxpy2", scope: !1, file: !1, line: 11, type: !7, isLocal: false, isDefinition: true, scopeLine: 11, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!22 = !DILocation(line: 13, column: 3, scope: !21)
!23 = !DILocation(line: 14, column: 17, scope: !21)
!24 = !DILocation(line: 14, column: 15, scope: !21)
!25 = !DILocation(line: 14, column: 5, scope: !21)
!26 = !DILocation(line: 14, column: 10, scope: !21)
!27 = distinct !{!27, !22, !28, !19, !29}
!28 = !DILocation(line: 14, column: 20, scope: !21)
!29 = !{!"tapir.loop.grainsize", i32 12}
!30 = !DILocation(line: 15, column: 1, scope: !21)
!31 = distinct !DISubprogram(name: "dsiny", scope: !1, file: !1, line: 18, type: !7, isLocal: false, isDefinition: true, scopeLine: 18, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!32 = !DILocation(line: 19, column: 3, scope: !31)
!33 = !DILocation(line: 20, column: 22, scope: !31)
!34 = !DILocation(line: 20, column: 34, scope: !31)
!35 = !DILocation(line: 20, column: 28, scope: !31)
!36 = !DILocation(line: 20, column: 5, scope: !31)
!37 = !DILocation(line: 21, column: 26, scope: !31)
!38 = !DILocation(line: 21, column: 29, scope: !31)
!39 = !DILocation(line: 21, column: 18, scope: !31)
!40 = !DILocation(line: 21, column: 14, scope: !31)
!41 = !DILocation(line: 21, column: 7, scope: !31)
!42 = !DILocation(line: 21, column: 12, scope: !31)
!43 = !DILocation(line: 20, column: 42, scope: !31)
!44 = distinct !{!44, !36, !38}
!45 = distinct !{!45, !32, !38, !19}
!46 = !DILocation(line: 22, column: 1, scope: !31)
!47 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 27, type: !7, isLocal: false, isDefinition: true, scopeLine: 27, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!48 = !DILocation(line: 29, column: 3, scope: !47)
!49 = !DILocation(line: 30, column: 5, scope: !47)
!50 = !DILocation(line: 31, column: 5, scope: !47)
!51 = !DILocation(line: 32, column: 9, scope: !47)
!52 = !DILocation(line: 32, column: 5, scope: !47)
!53 = !DILocation(line: 33, column: 3, scope: !47)
!54 = distinct !{!54, !48, !53, !19, !55}
!55 = !{!"tapir.loop.grainsize", i32 32}
!56 = !DILocation(line: 34, column: 1, scope: !47)
