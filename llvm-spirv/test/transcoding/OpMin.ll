; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: 4 TypeInt [[IntTypeID:[0-9]+]] 32 {{[0-9]+}}
; CHECK-SPIRV: 4 TypeVector [[Int2TypeID:[0-9]+]] [[IntTypeID]] 2
; CHECK-SPIRV: 6 CompositeInsert [[Int2TypeID]] [[CompositeID:[0-9]+]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV: 7 VectorShuffle [[Int2TypeID]] [[ShuffleID:[0-9]+]] [[CompositeID]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV: 7 ExtInst [[Int2TypeID]] {{[0-9]+}} 1 s_min {{[0-9]+}} [[ShuffleID]]

; CHECK-LLVM: call spir_func <2 x i32> @_Z3minDv2_iS_(

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @test() #0 !kernel_arg_addr_space !0 !kernel_arg_access_qual !0 !kernel_arg_type !0 !kernel_arg_base_type !0 !kernel_arg_type_qual !0 {
entry:
  %call = tail call spir_func <2 x i32> @_Z3minDv2_ii(<2 x i32> <i32 1, i32 10>, i32 5) #2
  ret void
}

declare spir_func <2 x i32> @_Z3minDv2_ii(<2 x i32>, i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!1}
!opencl.ocl.version = !{!2}
!opencl.used.extensions = !{!0}
!opencl.used.optional.core.features = !{!0}
!opencl.compiler.options = !{!0}

!0 = !{}
!1 = !{i32 1, i32 2}
!2 = !{i32 2, i32 0}
