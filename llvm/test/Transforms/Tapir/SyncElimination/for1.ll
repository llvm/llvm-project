; RUN: opt < %s -sync-elimination -S | FileCheck %s

; ModuleID = 'for1.cpp'
source_filename = "for1.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define void @_Z4funcv() #0 {
entry:
  %syncreg = call token @llvm.syncregion.start()
  %__init = alloca i32, align 4
  %__begin = alloca i32, align 4
  %__end = alloca i32, align 4
  %syncreg1 = call token @llvm.syncregion.start()
  %__init2 = alloca i32, align 4
  %__begin3 = alloca i32, align 4
  %__end4 = alloca i32, align 4
  store i32 0, i32* %__init, align 4
  store i32 0, i32* %__begin, align 4
  store i32 10, i32* %__end, align 4
  br label %pfor.cond

pfor.cond:                                        ; preds = %pfor.inc, %entry
  %0 = load i32, i32* %__begin, align 4
  %1 = load i32, i32* %__end, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %pfor.detach, label %pfor.end

pfor.detach:                                      ; preds = %pfor.cond
  %2 = load i32, i32* %__init, align 4
  %3 = load i32, i32* %__begin, align 4
  %mul = mul nsw i32 %3, 1
  %add = add nsw i32 %2, %mul
  detach within %syncreg, label %pfor.body.entry, label %pfor.inc

pfor.body.entry:                                  ; preds = %pfor.detach
  %i = alloca i32, align 4
  store i32 %add, i32* %i, align 4
  br label %pfor.body

pfor.body:                                        ; preds = %pfor.body.entry
  br label %pfor.preattach

pfor.preattach:                                   ; preds = %pfor.body
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.preattach, %pfor.detach
  %4 = load i32, i32* %__begin, align 4
  %inc = add nsw i32 %4, 1
  store i32 %inc, i32* %__begin, align 4
  br label %pfor.cond, !llvm.loop !1

pfor.end:                                         ; preds = %pfor.cond
  sync within %syncreg, label %pfor.end.continue

pfor.end.continue:                                ; preds = %pfor.end
  store i32 0, i32* %__init2, align 4
  store i32 0, i32* %__begin3, align 4
  store i32 10, i32* %__end4, align 4
  br label %pfor.cond3

; CHECK: pfor.end
; CHECK-NOT: sync
; CHECK: pfor.cond

pfor.cond3:                                       ; preds = %pfor.inc8, %pfor.end.continue
  %5 = load i32, i32* %__begin3, align 4
  %6 = load i32, i32* %__end4, align 4
  %cmp6 = icmp slt i32 %5, %6
  br i1 %cmp6, label %pfor.detach5, label %pfor.end10

pfor.detach5:                                     ; preds = %pfor.cond3
  %7 = load i32, i32* %__init2, align 4
  %8 = load i32, i32* %__begin3, align 4
  %mul8 = mul nsw i32 %8, 1
  %add9 = add nsw i32 %7, %mul8
  detach within %syncreg1, label %pfor.body.entry6, label %pfor.inc8

pfor.body.entry6:                                ; preds = %pfor.detach5
  %i11 = alloca i32, align 4
  store i32 %add9, i32* %i11, align 4
  br label %pfor.body6

pfor.body6:                                       ; preds = %pfor.body.entry5
  br label %pfor.preattach7

pfor.preattach7:                                  ; preds = %pfor.body6
  reattach within %syncreg1, label %pfor.inc8

pfor.inc8:                                        ; preds = %pfor.preattach7, %pfor.detach5
  %9 = load i32, i32* %__begin3, align 4
  %inc15 = add nsw i32 %9, 1
  store i32 %inc15, i32* %__begin3, align 4
  br label %pfor.cond3, !llvm.loop !3

pfor.end10:                                       ; preds = %pfor.cond3
  sync within %syncreg1, label %pfor.end.continue11

pfor.end.continue11:                              ; preds = %pfor.end10
  ret void
}

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }

!1 = distinct !{!1, !2}
!2 = !{!"tapir.loop.spawn.strategy", i32 1}
!3 = distinct !{!3, !2}
