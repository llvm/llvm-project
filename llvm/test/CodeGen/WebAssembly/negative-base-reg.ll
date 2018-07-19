; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32---elf"

@args = hidden local_unnamed_addr global [32 x i32] zeroinitializer, align 16

; Function Attrs: norecurse nounwind
define hidden i32 @main() local_unnamed_addr #0 {

; If LSR stops selecting a negative base reg value, then this test will no
; longer be useful as written.
; CHECK: i32.const $0=, -128
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.04 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
; The offset should not be folded into the store.
; CHECK: i32.const $push{{[0-9]+}}=, args+128
; CHECK: i32.add
; CHECK: i32.store 0(
  %arrayidx = getelementptr inbounds [32 x i32], [32 x i32]* @args, i32 0, i32 %i.04
  store i32 1, i32* %arrayidx, align 4, !tbaa !1
  %inc = add nuw nsw i32 %i.04, 1
  %exitcond = icmp eq i32 %inc, 32
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !5

for.end:                                          ; preds = %for.body
  ret i32 0
}

attributes #0 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 4.0.0 (trunk 279056) (llvm/trunk 279074)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.unroll.disable"}
