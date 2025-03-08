; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-EXT-OFF
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_fp_fast_math_mode %s -o - | FileCheck %s --check-prefix=CHECK-EXT-ON
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_fp_fast_math_mode %s -o - -filetype=obj | spirv-val %}

; CHECK-EXT-ON: OpCapability FPFastMathModeINTEL
; CHECK-EXT-ON: SPV_INTEL_fp_fast_math_mode
; CHECK-EXT-ON: OpName %[[#mul:]] "mul"
; CHECK-EXT-ON: OpName %[[#sub:]] "sub"
; CHECK-EXT-ON: OpDecorate %[[#mu:]] FPFastMathMode AllowContract
; CHECK-EXT-ON: OpDecorate %[[#su:]] FPFastMathMode AllowReassoc

; CHECK-EXT-OFF-NOT: OpCapability FPFastMathModeINTEL
; CHECK-EXT-OFF-NOT: SPV_INTEL_fp_fast_math_mode
; CHECK-EXT-OFF: OpName %[[#mul:]] "mul"
; CHECK-EXT-OFF: OpName %[[#sub:]] "sub"
; CHECK-EXT-OFF-NOT: 4 Decorate %[[#mul]] FPFastMathMode AllowContract
; CHECK-EXT-OFF-NOT: 4 Decorate %[[#sub]] FPFastMathMode AllowReassoc


; Function Attrs: convergent noinline norecurse nounwind optnone
define spir_kernel void @test(float %a, float %b) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %a.addr = alloca float, align 4
  %b.addr = alloca float, align 4
  store float %a, ptr %a.addr, align 4
  store float %b, ptr %b.addr, align 4
  %0 = load float, ptr %a.addr, align 4
  %1 = load float, ptr %a.addr, align 4
  %mul = fmul contract float %0, %1
  store float %mul, ptr %b.addr, align 4
  %2 = load float, ptr %b.addr, align 4
  %3 = load float, ptr %b.addr, align 4
  %sub = fsub reassoc float %2, %3
  store float %sub, ptr %b.addr, align 4
  ret void
}

attributes #0 = { convergent noinline norecurse nounwind optnone "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 12.0.0 (https://github.com/intel/llvm.git 5cf8088c994778561c8584d5433d7d32618725b2)"}
!3 = !{i32 0, i32 0}
!4 = !{!"none", !"none"}
!5 = !{!"float", !"float"}
!6 = !{!"", !""}

