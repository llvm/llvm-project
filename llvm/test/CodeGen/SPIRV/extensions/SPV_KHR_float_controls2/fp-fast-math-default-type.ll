; Check that FPFastMathDefault flags are applied to float arithmetic, float
; comparisons and calls returning float, but not to calls returning non-float
; types.

; RUN: llc -verify-machineinstrs -O2 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - | FileCheck %s --implicit-check-not=FPFastMathMode
; RUN: %if spirv-tools %{ llc -O2 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpDecorate %[[#MUL1:]] FPFastMathMode NotNaN|NotInf
; CHECK-DAG: OpDecorate %[[#MUL2:]] FPFastMathMode NotNaN|NotInf
; CHECK-DAG: OpDecorate %[[#MUL3:]] FPFastMathMode NotNaN|NotInf
; CHECK-DAG: OpDecorate %[[#CMP:]] FPFastMathMode NotNaN|NotInf
; CHECK-DAG: OpDecorate %[[#FCALL:]] FPFastMathMode NotNaN|NotInf
; CHECK-NOT: FPFastMathMode

; CHECK: %[[#MUL1]] = OpFMul
; CHECK: %[[#MUL2]] = OpFMul %[[#]] %[[#MUL1]]
; CHECK: %[[#MUL3]] = OpFMul
; CHECK: %[[#CMP]] = OpFOrdLessThan %[[#]] %[[#MUL3]]
; CHECK: %[[#FCALL]] = OpFunctionCall
; CHECK: OpFunctionCall
; CHECK: OpFunctionCall

define spir_kernel void @arith(ptr addrspace(1) %p, float %x, float %y, i1 %c) {
entry:
  br i1 %c, label %then, label %exit

then:
  %a = fmul float %x, %y
  store float %a, ptr addrspace(1) %p
  br label %exit

exit:
  %b = fmul float %x, %y
  %d = fmul float %b, %y
  store float %d, ptr addrspace(1) %p
  ret void
}

define spir_kernel void @relational(ptr addrspace(1) %p, ptr addrspace(1) %q, float %x, float %y, i1 %c) {
entry:
  br i1 %c, label %then, label %exit

then:
  %a = fmul float %x, %y
  store float %a, ptr addrspace(1) %p
  br label %exit

exit:
  %b = fmul float %x, %y
  %d = fcmp olt float %b, %y
  %e = zext i1 %d to i32
  store i32 %e, ptr addrspace(1) %q
  ret void
}

define spir_func float @float_callee(float %x) noinline {
  %r = fadd float %x, 1.0
  ret float %r
}

define spir_kernel void @float_call(ptr addrspace(1) %p, float %a) {
  %r = call spir_func float @float_callee(float %a)
  store float %r, ptr addrspace(1) %p
  ret void
}

define spir_func i32 @callee(i32 %x) noinline {
  %r = add i32 %x, 1
  ret i32 %r
}

define spir_kernel void @call(ptr addrspace(1) %p, i32 %a) {
  %r = call spir_func i32 @callee(i32 %a)
  store i32 %r, ptr addrspace(1) %p
  ret void
}

define spir_func i1 @bool_callee(i32 %x) noinline {
  %r = icmp eq i32 %x, 0
  ret i1 %r
}

define spir_kernel void @bool_call(ptr addrspace(1) %p, i32 %a) {
  %b = call spir_func i1 @bool_callee(i32 %a)
  %r = zext i1 %b to i32
  store i32 %r, ptr addrspace(1) %p
  ret void
}

!spirv.ExecutionMode = !{!0, !1, !2, !3, !4}
!0 = !{ptr @arith, i32 6028, float poison, i32 3}
!1 = !{ptr @relational, i32 6028, float poison, i32 3}
!2 = !{ptr @call, i32 6028, float poison, i32 3}
!3 = !{ptr @bool_call, i32 6028, float poison, i32 3}
!4 = !{ptr @float_call, i32 6028, float poison, i32 3}
