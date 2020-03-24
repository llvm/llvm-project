; RUN: opt < %s -licm -S | FileCheck %s

; Function Attrs: nounwind uwtable
define i32 @main() #0 {
entry:
; CHECK-NOT: call i64 @strlen
  %A = alloca [1000 x i8], align 16
  %syncreg = tail call token @llvm.syncregion.start()
  %0 = getelementptr inbounds [1000 x i8], [1000 x i8]* %A, i64 0, i64 0
  call void @llvm.lifetime.start.p0i8(i64 1000, i8* nonnull %0) #3
  br label %pfor.detach

pfor.cond.cleanup:                                ; preds = %pfor.inc
  sync within %syncreg, label %pfor.end.continue

pfor.end.continue:                                ; preds = %pfor.cond.cleanup
  %1 = load i8, i8* %0, align 16, !tbaa !2
  %conv2 = sext i8 %1 to i32
  %arrayidx3 = getelementptr inbounds [1000 x i8], [1000 x i8]* %A, i64 0, i64 999
  %2 = load i8, i8* %arrayidx3, align 1, !tbaa !2
  %conv4 = sext i8 %2 to i32
  %add5 = add nsw i32 %conv4, %conv2
  call void @llvm.lifetime.end.p0i8(i64 1000, i8* nonnull %0) #3
  ret i32 %add5

pfor.detach:                                      ; preds = %pfor.inc, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %pfor.inc ]
  detach within %syncreg, label %pfor.body, label %pfor.inc

pfor.body:                                        ; preds = %pfor.detach
; CHECK: call i64 @strlen
  %call = call i64 @strlen(i8* nonnull %0)
  %conv = trunc i64 %call to i8
  %arrayidx = getelementptr inbounds [1000 x i8], [1000 x i8]* %A, i64 0, i64 %indvars.iv
  store i8 %conv, i8* %arrayidx, align 1, !tbaa !2
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.body, %pfor.detach
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %pfor.cond.cleanup, label %pfor.detach, !llvm.loop !5
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

; Function Attrs: argmemonly nounwind readonly
declare i64 @strlen(i8* nocapture) local_unnamed_addr #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { argmemonly nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (https://github.com/wsmoses/Cilk-Clang a74c03783b70009d74a58b002db5233635fc7e15) (git@github.com:wsmoses/Parallel-IR fc410c92d294d90b54ffd4bc7f3f11dffef9ac1e)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = distinct !{!5, !6}
!6 = !{!"tapir.loop.spawn.strategy", i32 1}
