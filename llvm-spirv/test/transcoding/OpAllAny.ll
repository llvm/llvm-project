; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-LLVM: call spir_func i32 @_Z3allDv2_i(
; CHECK-LLVM: call spir_func i32 @_Z3anyDv2_i(

; CHECK-SPIRV: 2 TypeBool [[BoolTypeID:[0-9]+]]
; CHECK-SPIRV: 4 All [[BoolTypeID]]
; CHECK-SPIRV: 4 Any [[BoolTypeID]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @testKernel() #0 !kernel_arg_addr_space !0 !kernel_arg_access_qual !0 !kernel_arg_type !0 !kernel_arg_base_type !0 !kernel_arg_type_qual !0 {
entry:
  %cmp = icmp ne <2 x i64> zeroinitializer, <i64 1, i64 1>
  %sext = sext <2 x i1> %cmp to <2 x i64>
  %call = call spir_func i32 @_Z3allDv2_l(<2 x i64> %sext)
  %0 = insertelement <2 x i64> <i64 1, i64 1>, i64 0, i32 0
  %cmp1 = icmp ne <2 x i64> zeroinitializer, %0
  %sext2 = sext <2 x i1> %cmp1 to <2 x i64>
  %call3 = call spir_func i32 @_Z3anyDv2_l(<2 x i64> %sext2)
  ret void
}

declare spir_func i32 @_Z3allDv2_l(<2 x i64>) #1

declare spir_func i32 @_Z3anyDv2_l(<2 x i64>) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!1}
!opencl.ocl.version = !{!2}
!opencl.used.extensions = !{!0}
!opencl.used.optional.core.features = !{!0}
!opencl.compiler.options = !{!0}

!0 = !{}
!1 = !{i32 1, i32 2}
!2 = !{i32 2, i32 0}
