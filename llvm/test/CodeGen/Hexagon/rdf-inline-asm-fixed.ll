; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: r0 = #24
; CHECK-NEXT: r1 =
; // R2 should be assigned a value from R3+.
; CHECK-NEXT: r2 = r{{[3-9]}}
; CHECK-NEXT: trap0

target datalayout = "e-m:e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a:0-n16:32"
target triple = "hexagon"

; Function Attrs: nounwind
define i32 @foo(i32 %status) #0 {
entry:
  %arg1 = alloca i32, align 4
  %0 = bitcast i32* %arg1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #2
  store i32 %status, i32* %arg1, align 4, !tbaa !1
  %1 = call i32 asm sideeffect "r0 = #$1\0Ar1 = $2\0Ar2 = $4\0Atrap0 (#0)\0A$0 = r0", "=r,i,r,*m,r,~{r0},~{r1},~{r2}"(i32 24, i32* nonnull %arg1, i32* nonnull %arg1, i32 %status) #2, !srcloc !5
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %0) #2
  ret i32 %1
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv5" "target-features"="-hvx,-hvx-double" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind }

!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{i32 110, i32 129, i32 146, i32 163, i32 183}
