; Test that Tapir's loop spawning pass transforms this simple loop
; into recursive divide-and-conquer.

; RUN: opt < %s -loop-spawning -S | FileCheck %s

; Function Attrs: nounwind uwtable
define void @foo(i32 %n) local_unnamed_addr #0 {
; CHECK-LABEL: @foo(
entry:
  %syncreg = call token @llvm.syncregion.start()
  %cmp5 = icmp sgt i32 %n, 0
  br i1 %cmp5, label %pfor.detach.preheader, label %pfor.cond.cleanup

pfor.detach.preheader:                            ; preds = %entry
; CHECK: pfor.detach.preheader:
; CHECK: [[LIMIT:%[0-9]+]] = add [[TYPE:i[0-9]+]] %n, -1
; CHECK: call fastcc void @[[OUTLINED:[a-zA-Z0-9._]+]](
; CHECK: [[TYPE]] 0
; CHECK: [[TYPE]] [[LIMIT]]
; CHECK: [[TYPE]] {{[%]?[a-zA-Z0-9._]+}}
; CHECK-NEXT: br label %pfor.cond.cleanup.loopexit
  br label %pfor.detach

pfor.cond.cleanup.loopexit:                       ; preds = %pfor.inc
  br label %pfor.cond.cleanup

pfor.cond.cleanup:                                ; preds = %pfor.cond.cleanup.loopexit, %entry
; CHECK: pfor.cond.cleanup
; CHECK-NOT: sync within %syncreg, label %0
  sync within %syncreg, label %0

; <label>:0:                                      ; preds = %pfor.cond.cleanup
  ret void

pfor.detach:                                      ; preds = %pfor.detach.preheader, %pfor.inc
; CHECK: pfor.detach:
; CHECK: phi i32
; CHECK-NOT: %pfor.detach.preheader
; CHECK: detach

; CHECK: define internal fastcc void @[[OUTLINED]](
; CHECK: [[TYPE]] [[START:%[a-zA-Z0-9._]+]]
; CHECK: [[TYPE]] [[END:%[a-zA-Z0-9._]+]]
; CHECK: [[TYPE]] [[GRAIN:%[a-zA-Z0-9._]+]]
; CHECK: [[NEWSYNCREG:%[a-zA-Z0-9._]+]] = call token @llvm.syncregion.start(

; CHECK: {{^(; <label>:)?}}[[DACSTART:[a-zA-Z0-9._]+]]:
; CHECK: [[ITERSTART:%[a-zA-Z0-9._]+]] = phi [[TYPE]] [{{.*}}[[START]]{{.*}}]
; CHECK-NEXT: [[ITERCOUNT:%[a-zA-Z0-9._]+]] = sub [[TYPE]] [[END]], [[ITERSTART]]
; CHECK-NEXT: [[CMP:%[0-9]+]] = icmp ugt [[TYPE]] [[ITERCOUNT]], [[GRAIN]]
; CHECK-NEXT: br i1 [[CMP]], label %[[RECUR:[0-9]+]], label %[[BODY:[0-9]+]]

; CHECK: {{^(; <label>:)?}}[[RECUR]]:
; CHECK-NEXT: [[HALFCOUNT:%[a-zA-Z0-9._]+]] = lshr [[TYPE]] [[ITERCOUNT]], 1
; CHECK-NEXT: [[MIDITER:%[a-zA-Z0-9._]+]] = add {{.*}} [[TYPE]] [[ITERSTART]], [[HALFCOUNT]]
; CHECK-NEXT: detach within [[NEWSYNCREG]], label %[[DETACHED:[a-zA-Z0-9._]+]], label %[[CONTINUE:[a-zA-Z0-9._]+]]

; CHECK: {{^(; <label>:)?}}[[DETACHED]]:
; CHECK-NEXT: call fastcc void @[[OUTLINED]]([[TYPE]] [[ITERSTART]], [[TYPE]] [[MIDITER]], [[TYPE]] [[GRAIN]]
; CHECK-NEXT: reattach within [[NEWSYNCREG]], label %[[CONTINUE]]

; CHECK: {{^(; <label>:)?}}[[CONTINUE]]:
; CHECK-NEXT: [[MIDITERP1:%[a-zA-Z0-9._]+]] = add {{.*}} [[TYPE]] [[MIDITER]], 1
; CHECK-NEXT: br label %[[DACSTART]]
  %i.06 = phi i32 [ %inc, %pfor.inc ], [ 0, %pfor.detach.preheader ]
  detach within %syncreg, label %pfor.body, label %pfor.inc
; CHECK: sync within [[NEWSYNCREG]]
; CHECK: br label %pfor.body.ls

pfor.body:                                        ; preds = %pfor.detach
; CHECK: pfor.body.ls:
  tail call void @bar(i32 %i.06) #2
; CHECK-NEXT: tail call void @bar(i32 %i.06.ls)
  reattach within %syncreg, label %pfor.inc
; CHECK-NEXT: br label %[[INC:[a-zA-Z0-9._]+]]

pfor.inc:                                         ; preds = %pfor.body, %pfor.detach
; CHECK: {{^(; <label>:)?}}[[INC]]:
; CHECK-NEXT: [[LOCALCMP:%[0-9]+]] = icmp ult {{.*}} [[LOCALITER:%[a-zA-Z0-9._]+]], [[END]]
  %inc = add nuw nsw i32 %i.06, 1
; CHECK-NEXT: add {{.*}} [[LOCALITER]], 1
  %exitcond = icmp eq i32 %inc, %n
; CHECK: br i1 [[LOCALCMP]]
  br i1 %exitcond, label %pfor.cond.cleanup.loopexit, label %pfor.detach, !llvm.loop !1
}

declare void @bar(i32) local_unnamed_addr #1

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #3

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { argmemonly nounwind }

!1 = distinct !{!1, !2}
!2 = !{!"tapir.loop.spawn.strategy", i32 1}
