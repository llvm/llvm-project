; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[TyInt:.*]] = OpTypeInt 8 0
; CHECK: %[[TyPtr:.*]] = OpTypePointer {{[a-zA-Z]+}} %[[TyInt]]
; CHECK: OpFunctionParameter %[[TyPtr]]
; CHECK: OpFunctionParameter %[[TyPtr]]

%struct.my_kernel_data = type { i32, i32, i32, i32, i32 }
%struct.my_struct = type { i32, i32 }

define spir_kernel void @test(ptr addrspace(1) %in, ptr addrspace(1) %outData) !kernel_arg_type !5 {
entry:
  ret void
}

!llvm.module.flags = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}
!opencl.used.extensions = !{!3}
!opencl.used.optional.core.features = !{!3}
!opencl.compiler.options = !{!3}
!llvm.ident = !{!4}
!opencl.kernels = !{!6}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
!3 = !{}
!4 = !{!"clang version 6.0.0"}
!5 = !{!"my_kernel_data*", !"struct my_struct*"}
!6 = !{ptr @test}

