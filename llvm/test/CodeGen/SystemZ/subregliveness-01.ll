; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 -verify-machineinstrs -systemz-subreg-liveness < %s | FileCheck %s

; Check for successful compilation.
; CHECK: lgfrl %r0, g_399

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-ibm-linux"

@g_439 = external global i32, align 4
@g_399 = external global { i8, i8, i8, i8, i8, i8 }, align 8

; Function Attrs: nounwind
define void @main() #0 {
bb:
  %tmp = load i48, i48* bitcast ({ i8, i8, i8, i8, i8, i8 }* @g_399 to i48*), align 8, !noalias !1
  %tmp1 = ashr i48 %tmp, 17
  %tmp2 = trunc i48 %tmp1 to i32
  %tmp3 = sext i32 %tmp2 to i64
  br label %bb4

bb4:                                              ; preds = %bb4, %bb
  %tmp5 = load i64, i64* undef, align 8, !tbaa !4, !noalias !1
  %tmp6 = urem i64 -923186811629238421, %tmp3
  %tmp7 = or i64 %tmp6, %tmp5
  %tmp8 = trunc i64 %tmp7 to i32
  %tmp9 = lshr i32 %tmp8, 2
  %tmp10 = and i32 %tmp9, 60
  %tmp11 = xor i32 %tmp10, -1592309976
  %tmp12 = or i32 0, %tmp11
  %tmp13 = or i32 %tmp12, 3
  store i32 %tmp13, i32* @g_439, align 4, !tbaa !8, !noalias !1
  br label %bb4
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="z13" "target-features"="+transactional-execution,+vector" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 8.0.0 (http://llvm.org/git/clang.git c0a5e830f198cf42d29f72f1ec06fbf4c5210e2c) (http://llvm.org/git/llvm.git ffc8c538b70b678031b8617f61f83ee120bcb884)"}
!1 = !{!2}
!2 = distinct !{!2, !3, !"func_1: %agg.result"}
!3 = distinct !{!3, !"func_1"}
!4 = !{!5, !5, i64 0}
!5 = !{!"long", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !6, i64 0}
