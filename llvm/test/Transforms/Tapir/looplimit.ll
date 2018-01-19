; Test that Tapir's loop spawning pass correctly transforms a loop
; that reads its original end iteration count.

; RUN: opt < %s -loop-spawning -S -ls-tapir-target=cilk | FileCheck %s

source_filename = "looplimittest.c"

@.str = private unnamed_addr constant [13 x i8] c"Limit is %d\0A\00", align 1
@str = private unnamed_addr constant [9 x i8] c"Starting\00"
@str.3 = private unnamed_addr constant [9 x i8] c"Finished\00"

; Function Attrs: noinline nounwind uwtable
define void @foo(i32 %limit) local_unnamed_addr #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp9 = icmp slt i32 %limit, 0
  br i1 %cmp9, label %pfor.cond.cleanup, label %pfor.detach

; CHECK: pfor.detach.preheader:
; CHECK: call fastcc void @[[OUTLINED:[a-zA-Z0-9._]+]](
; CHECK: [[TYPE:i[0-9]+]] 0
; CHECK: [[TYPE]] [[LOOPLIMIT:%[a-zA-Z0-9._]+]]
; CHECK: [[TYPE]] {{[%]?[a-zA-Z0-9._]+}}
; CHECK: i32 %limit

pfor.cond.cleanup:                                ; preds = %pfor.inc, %entry
  sync within %syncreg, label %pfor.end.continue

pfor.end.continue:                                ; preds = %pfor.cond.cleanup
  ret void

; CHECK: define internal fastcc void @[[OUTLINED]](
; CHECK: [[TYPE]] [[START:%[a-zA-Z0-9._]+]]
; CHECK: [[TYPE]] [[END:%[a-zA-Z0-9._]+]]
; CHECK: [[TYPE]] [[GRAIN:%[a-zA-Z0-9._]+]]
; CHECK: i32 [[LIMITARG:%[a-zA-Z0-9._]+]]

; CHECK: [[NEWSYNCREG:%[a-zA-Z0-9._]+]] = tail call token @llvm.syncregion.start(

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
; CHECK-NEXT: call fastcc void @[[OUTLINED]]([[TYPE]] [[ITERSTART]], [[TYPE]] [[MIDITER]], [[TYPE]] [[GRAIN]], i32 [[LIMITARG]]
; CHECK-NEXT: reattach within [[NEWSYNCREG]], label %[[CONTINUE]]

; CHECK: {{^(; <label>:)?}}[[CONTINUE]]:
; CHECK-NEXT: [[MIDITERP1:%[a-zA-Z0-9._]+]] = add {{.*}} [[TYPE]] [[MIDITER]], 1
; CHECK-NEXT: br label %[[DACSTART]]

pfor.detach:                                      ; preds = %entry, %pfor.inc
  %__begin.010 = phi i32 [ %inc, %pfor.inc ], [ 0, %entry ]
  detach within %syncreg, label %pfor.body, label %pfor.inc

pfor.body:                                        ; preds = %pfor.detach
; CHECK: {{^(; <label>:)?}}[[BODY]]:
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i64 0, i64 0), i32 %limit)
; CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i64 0, i64 0), i32 [[LIMITARG]])
  reattach within %syncreg, label %pfor.inc
; CHECK: br label %[[INC:[a-zA-Z0-9._]+]]

pfor.inc:                                         ; preds = %pfor.body, %pfor.detach
; CHECK: {{^(; <label>:)?}}[[INC]]:
; CHECK-NEXT: [[LOCALCMP:%[0-9]+]] = icmp ult {{.*}} [[LOCALITER:%[a-zA-Z0-9._]+]], [[END]]
  %inc = add nuw nsw i32 %__begin.010, 1
; CHECK-NEXT: add {{.*}} [[LOCALITER]], 1
  %exitcond = icmp eq i32 %__begin.010, %limit
; CHECK: br i1 [[LOCALCMP]]
  br i1 %exitcond, label %pfor.cond.cleanup, label %pfor.detach, !llvm.loop !2
}

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #2

; Function Attrs: nounwind
declare i32 @puts(i8* nocapture readonly) #4

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!2 = distinct !{!2, !3}
!3 = !{!"tapir.loop.spawn.strategy", i32 1}
