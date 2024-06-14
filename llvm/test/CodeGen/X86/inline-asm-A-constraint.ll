; RUN: llc -mtriple=x86_64-- < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-- -early-live-intervals < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--"

; Function Attrs: nounwind uwtable
define { i64, i64 } @foo(ptr %ptr, ptr nocapture readonly %src, ptr nocapture readonly %dst) local_unnamed_addr #0 {
entry:
  %0 = load i128, ptr %dst, align 16, !tbaa !1
  %shr = lshr i128 %0, 64
  %conv = trunc i128 %shr to i64
  %conv1 = trunc i128 %0 to i64
  %1 = load i128, ptr %src, align 16, !tbaa !1
  %2 = tail call i128 asm sideeffect "lock; cmpxchg16b $1", "=A,=*m,{cx},{bx},0,*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i8) %ptr, i64 %conv, i64 %conv1, i128 %1, ptr elementtype(i8) %ptr) #1, !srcloc !5
  %retval.sroa.0.0.extract.trunc = trunc i128 %2 to i64
  %retval.sroa.2.0.extract.shift = lshr i128 %2, 64
  %retval.sroa.2.0.extract.trunc = trunc i128 %retval.sroa.2.0.extract.shift to i64
  %.fca.0.insert = insertvalue { i64, i64 } undef, i64 %retval.sroa.0.0.extract.trunc, 0
  %.fca.1.insert = insertvalue { i64, i64 } %.fca.0.insert, i64 %retval.sroa.2.0.extract.trunc, 1
  ret { i64, i64 } %.fca.1.insert
}
; CHECK: lock
; CHECK-NEXT: cmpxchg16b

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 5.0.0 (trunk 300088)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"__int128", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{i32 269}
