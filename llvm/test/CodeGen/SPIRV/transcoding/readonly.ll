; RUN: llc -O0 %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpDecorate %[[PARAM:[0-9]+]] FuncParamAttr NoWrite
; CHECK-SPIRV: %[[PARAM]] = OpFunctionParameter %{{.*}}

; ModuleID = 'readonly.bc'
source_filename = "readonly.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spirv64-unknown-unknown"

; Function Attrs: norecurse nounwind readonly willreturn
define dso_local spir_kernel void @_ZTSZ4mainE15kernel_function(i32 addrspace(1)* readonly %_arg_) local_unnamed_addr #0 {
entry:
  ret void
}

attributes #0 = { norecurse nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 13.0.0 (https://github.com/intel/llvm.git)"}
