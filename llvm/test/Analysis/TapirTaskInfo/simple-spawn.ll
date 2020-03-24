; RUN: opt < %s -analyze -tasks | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
; Function Attrs: nounwind uwtable
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

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

declare void @bar(i32*) local_unnamed_addr #2

declare void @baz(i32*, i32*) local_unnamed_addr #2

; CHECK: task at depth 0 containing: <task entry><func sp entry>%entry<sp exit><phi sp entry>%det.cont<sp exit><sync sp entry>%sync.continue<sp exit><task exit>
; CHECK: task at depth 1 containing: <task entry><task sp entry>%det.achd<sp exit><task exit>

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

