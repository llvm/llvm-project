; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: 4 TypeInt [[IntTypeID:[0-9]+]] 64
; CHECK-SPIRV: 4 TypeVector [[Int64VectorTypeID:[0-9]+]] [[IntTypeID]] 2

; CHECK-SPIRV: 6 Select [[Int64VectorTypeID]]
; CHECK-SPIRV: 6 Select [[Int64VectorTypeID]]
; CHECK-SPIRV: 6 Select [[Int64VectorTypeID]]
; CHECK-SPIRV: 6 Select [[Int64VectorTypeID]]

; CHECK-LLVM: call spir_func <2 x i64> @_Z5isinfDv2_d(<2 x double>
; CHECK-LLVM: call spir_func <2 x i64> @_Z5isnanDv2_d(<2 x double>
; CHECK-LLVM: call spir_func <2 x i64> @_Z8isnormalDv2_d(<2 x double>
; CHECK-LLVM: call spir_func <2 x i64> @_Z8isfiniteDv2_d(<2 x double>

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @test_vector_double(<2 x i64> addrspace(1)* nocapture %out, <2 x double> %in) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
  %1 = tail call spir_func <2 x i64> @_Z5isinfDv2_d(<2 x double> %in) #2
  %2 = tail call spir_func <2 x i64> @_Z5isnanDv2_d(<2 x double> %in) #2
  %3 = add <2 x i64> %1, %2
  %4 = tail call spir_func <2 x i64> @_Z8isnormalDv2_d(<2 x double> %in) #2
  %5 = add <2 x i64> %3, %4
  %6 = tail call spir_func <2 x i64> @_Z8isfiniteDv2_d(<2 x double> %in) #2
  %7 = add <2 x i64> %5, %6
  store <2 x i64> %7, <2 x i64> addrspace(1)* %out, align 16, !tbaa !11
  ret void
}

declare spir_func <2 x i64> @_Z5isinfDv2_d(<2 x double>) #1

declare spir_func <2 x i64> @_Z5isnanDv2_d(<2 x double>) #1

declare spir_func <2 x i64> @_Z8isnormalDv2_d(<2 x double>) #1

declare spir_func <2 x i64> @_Z8isfiniteDv2_d(<2 x double>) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!9}
!opencl.compiler.options = !{!8}
!llvm.ident = !{!10}

!1 = !{i32 1, i32 0}
!2 = !{!"none", !"none"}
!3 = !{!"long2*", !"double2"}
!4 = !{!"long2*", !"double2"}
!5 = !{!"", !""}
!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 0}
!8 = !{}
!9 = !{!"cl_doubles"}
!10 = !{!"clang version 3.6.1 "}
!11 = !{!12, !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C/C++ TBAA"}
