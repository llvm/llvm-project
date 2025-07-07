; RUN: llc -mtriple=mipsel-linux-gnu -mattr=mips16 -relocation-model=static < %s | FileCheck %s

declare float     @llvm.powi.f32.i32(float  %Val, i32 %power)
declare double    @llvm.powi.f64.i32(double %Val, i32 %power)

define float @foo_pow_f32(float %y, i32 %p)  {
  %1 = tail call float @llvm.powi.f32.i32(float %y, i32 %p)
; CHECK-NOT: .ent	__call_stub_fp_llvm.powi.f32
; CHECK-NOT: {{.*}} jal llvm.powi.f32
  ret float %1
} 

define double @foo_pow_f64(double %y, i32 %p)  {
  %1 = tail call double @llvm.powi.f64.i32(double %y, i32 %p)
; CHECK-NOT: .ent	__call_stub_fp_llvm.powi.f64
; CHECK-NOT: {{.*}} jal llvm.powi.f64 
  ret double %1
} 

attributes #0 = { nounwind optsize "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #1 = { nounwind readonly }

!0 = !{!"double", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!"int", !1}
