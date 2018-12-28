; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-LLVM: call spir_func i32 @_Z8isfinitef(
; CHECK-LLVM: call spir_func i32 @_Z5isnanf(
; CHECK-LLVM: call spir_func i32 @_Z5isinff(
; CHECK-LLVM: call spir_func i32 @_Z8isnormalf(
; CHECK-LLVM: call spir_func i32 @_Z7signbitf(

; CHECK-LLVM: call spir_func <2 x i32> @_Z8isfiniteDv2_f(
; CHECK-LLVM: call spir_func <2 x i32> @_Z5isnanDv2_f(
; CHECK-LLVM: call spir_func <2 x i32> @_Z5isinfDv2_f(
; CHECK-LLVM: call spir_func <2 x i32> @_Z8isnormalDv2_f(

; CHECK-SPIRV: 2 TypeBool [[BoolTypeID:[0-9]+]]
; CHECK-SPIRV: 4 TypeVector [[BoolVectorTypeID:[0-9]+]] [[BoolTypeID]] 2

; CHECK-SPIRV: 4 IsFinite [[BoolTypeID]]
; CHECK-SPIRV: 4 IsNan [[BoolTypeID]]
; CHECK-SPIRV: 4 IsInf [[BoolTypeID]]
; CHECK-SPIRV: 4 IsNormal [[BoolTypeID]]
; CHECK-SPIRV: 4 SignBitSet [[BoolTypeID]]

; CHECK-SPIRV: 4 IsFinite [[BoolVectorTypeID]]
; CHECK-SPIRV: 4 IsNan [[BoolVectorTypeID]]
; CHECK-SPIRV: 4 IsInf [[BoolVectorTypeID]]
; CHECK-SPIRV: 4 IsNormal [[BoolVectorTypeID]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @test_scalar(i32 addrspace(1)* nocapture %out, float %f) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %call = tail call spir_func i32 @_Z8isfinitef(float %f) #2
  %call1 = tail call spir_func i32 @_Z5isnanf(float %f) #2
  %add = add nsw i32 %call1, %call
  %call2 = tail call spir_func i32 @_Z5isinff(float %f) #2
  %add3 = add nsw i32 %add, %call2
  %call4 = tail call spir_func i32 @_Z8isnormalf(float %f) #2
  %add5 = add nsw i32 %add3, %call4
  %call6 = tail call spir_func i32 @_Z7signbitf(float %f) #2
  %add7 = add nsw i32 %add5, %call6
  store i32 %add7, i32 addrspace(1)* %out, align 4
  ret void
}

declare spir_func i32 @_Z8isfinitef(float) #1

declare spir_func i32 @_Z5isnanf(float) #1

declare spir_func i32 @_Z5isinff(float) #1

declare spir_func i32 @_Z8isnormalf(float) #1

declare spir_func i32 @_Z7signbitf(float) #1

; Function Attrs: nounwind
define spir_kernel void @test_vector(<2 x i32> addrspace(1)* nocapture %out, <2 x float> %f) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !7 !kernel_arg_base_type !8 !kernel_arg_type_qual !5 {
entry:
  %call = tail call spir_func <2 x i32> @_Z8isfiniteDv2_f(<2 x float> %f) #2
  %call1 = tail call spir_func <2 x i32> @_Z5isnanDv2_f(<2 x float> %f) #2
  %add = add <2 x i32> %call, %call1
  %call2 = tail call spir_func <2 x i32> @_Z5isinfDv2_f(<2 x float> %f) #2
  %add3 = add <2 x i32> %add, %call2
  %call4 = tail call spir_func <2 x i32> @_Z8isnormalDv2_f(<2 x float> %f) #2
  %add5 = add <2 x i32> %add3, %call4
  store <2 x i32> %add5, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

declare spir_func <2 x i32> @_Z8isfiniteDv2_f(<2 x float>) #1

declare spir_func <2 x i32> @_Z5isnanDv2_f(<2 x float>) #1

declare spir_func <2 x i32> @_Z5isinfDv2_f(<2 x float>) #1

declare spir_func <2 x i32> @_Z8isnormalDv2_f(<2 x float>) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!9}
!opencl.ocl.version = !{!10}
!opencl.used.extensions = !{!11}
!opencl.used.optional.core.features = !{!11}
!opencl.compiler.options = !{!11}

!1 = !{i32 1, i32 0}
!2 = !{!"none", !"none"}
!3 = !{!"int*", !"float"}
!4 = !{!"int*", !"float"}
!5 = !{!"", !""}
!7 = !{!"int2*", !"float2"}
!8 = !{!"int2*", !"float2"}
!9 = !{i32 1, i32 2}
!10 = !{i32 2, i32 0}
!11 = !{}
