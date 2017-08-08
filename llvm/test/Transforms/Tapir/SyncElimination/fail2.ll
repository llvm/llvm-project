; RUN: opt < %s -sync-elimination -S | FileCheck %s

; ModuleID = 'fail2.cpp'
source_filename = "fail2.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define void @_Z4funcPiS_(i32* %a, i32* %b) #0 {
entry:
  %a.addr = alloca i32*, align 8
  %b.addr = alloca i32*, align 8
  %syncreg = call token @llvm.syncregion.start()
  store i32* %a, i32** %a.addr, align 8
  store i32* %b, i32** %b.addr, align 8
  detach within %syncreg, label %det.achd, label %det.cont

det.achd:                                         ; preds = %entry
  %0 = load i32*, i32** %a.addr, align 8
  store i32 1, i32* %0, align 4
  reattach within %syncreg, label %det.cont

det.cont:                                         ; preds = %det.achd, %entry
  sync within %syncreg, label %sync.continue
; CHECK: sync within %syncreg, label %sync.continue

sync.continue:                                    ; preds = %det.cont
  %1 = load i32*, i32** %b.addr, align 8
  store i32 2, i32* %1, align 4
  sync within %syncreg, label %sync.continue1
; CHECK-NOT: sync within %syncreg, label %sync.continue1

sync.continue1:                                   ; preds = %sync.continue
  ret void
; CHECK: ret void
}

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
