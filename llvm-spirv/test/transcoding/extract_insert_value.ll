; ModuleID = ''
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

; Check 'LLVM ==> SPIR-V ==> LLVM' conversion of extractvalue/insertvalue.

%struct.arr = type { [7 x float] }
%struct.st = type { %struct.inner }
%struct.inner = type { float }
; CHECK-LLVM: %struct.arr = type { [7 x float] }
; CHECK-LLVM: %struct.st = type { %struct.inner }
; CHECK-LLVM: %struct.inner = type { float }

; CHECK-LLVM:         define spir_func void @array_test
; CHECK-LLVM-LABEL:   entry
; CHECK-LLVM:         %0 = getelementptr inbounds %struct.arr, %struct.arr addrspace(1)* %object, i32 0, i32 0
; CHECK-LLVM:         %1 = load [7 x float], [7 x float] addrspace(1)* %0, align 4
; CHECK-LLVM:         %2 = extractvalue [7 x float] %1, 4
; CHECK-LLVM:         %3 = extractvalue [7 x float] %1, 2
; CHECK-LLVM:         %4 = fadd float %2, %3
; CHECK-LLVM:         %5 = insertvalue [7 x float] %1, float %4, 5
; CHECK-LLVM:         store [7 x float] %5, [7 x float] addrspace(1)* %0

; CHECK-SPIRV-LABEL:  5 Function
; CHECK-SPIRV-NEXT:   FunctionParameter {{[0-9]+}} [[object:[0-9]+]]
; CHECK-SPIRV:        6 InBoundsPtrAccessChain {{[0-9]+}} {{[0-9]+}} [[object]] {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV:        6 Load {{[0-9]+}} [[extracted_array:[0-9]+]] {{[0-9]+}} {{[0-9]+}} 4
; CHECK-SPIRV:        5 CompositeExtract {{[0-9]+}} [[elem_4:[0-9]+]] [[extracted_array]] 4
; CHECK-SPIRV:        5 CompositeExtract {{[0-9]+}} [[elem_2:[0-9]+]] [[extracted_array]] 2
; CHECK-SPIRV:        5 FAdd {{[0-9]+}} [[add:[0-9]+]] [[elem_4]] [[elem_2]]
; CHECK-SPIRV:        6 CompositeInsert {{[0-9]+}} [[inserted_array:[0-9]+]] [[add]] [[extracted_array]] 5
; CHECK-SPIRV:        3 Store {{[0-9]+}} [[inserted_array]]
; CHECK-SPIRV-LABEL:  1 FunctionEnd

; Function Attrs: nounwind
define spir_func void @array_test(%struct.arr addrspace(1)* %object) #0 {
entry:
  %0 = getelementptr inbounds %struct.arr, %struct.arr addrspace(1)* %object, i32 0, i32 0
  %1 = load [7 x float], [7 x float] addrspace(1)* %0, align 4
  %2 = extractvalue [7 x float] %1, 4
  %3 = extractvalue [7 x float] %1, 2
  %4 = fadd float %2, %3
  %5 = insertvalue [7 x float] %1, float %4, 5
  store [7 x float] %5, [7 x float] addrspace(1)* %0
  ret void
}

; CHECK-LLVM:         define spir_func void @struct_test
; CHECK-LLVM-LABEL:   entry
; CHECK-LLVM:         %0 = getelementptr inbounds %struct.st, %struct.st addrspace(1)* %object, i32 0, i32 0
; CHECK-LLVM:         %1 = load %struct.inner, %struct.inner addrspace(1)* %0, align 4
; CHECK-LLVM:         %2 = extractvalue %struct.inner %1, 0
; CHECK-LLVM:         %3 = fadd float %2, 1.000000e+00
; CHECK-LLVM:         %4 = insertvalue %struct.inner %1, float %3, 0
; CHECK-LLVM:         store %struct.inner %4, %struct.inner addrspace(1)* %0

; CHECK-SPIRV-LABEL:  5 Function
; CHECK-SPIRV-NEXT:   FunctionParameter {{[0-9]+}} [[object:[0-9]+]]
; CHECK-SPIRV:        6 InBoundsPtrAccessChain {{[0-9]+}} {{[0-9]+}} [[object]] {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV:        6 Load {{[0-9]+}} [[extracted_struct:[0-9]+]] {{[0-9]+}} {{[0-9]+}} 4
; CHECK-SPIRV:        5 CompositeExtract {{[0-9]+}} [[elem:[0-9]+]] [[extracted_struct]] 0
; CHECK-SPIRV:        5 FAdd {{[0-9]+}} [[add:[0-9]+]] [[elem]] {{[0-9]+}}
; CHECK-SPIRV:        6 CompositeInsert {{[0-9]+}} [[inserted_struct:[0-9]+]] [[add]] [[extracted_struct]] 0
; CHECK-SPIRV:        3 Store {{[0-9]+}} [[inserted_struct]]
; CHECK-SPIRV-LABEL:  1 FunctionEnd

; Function Attrs: nounwind
define spir_func void @struct_test(%struct.st addrspace(1)* %object) #0 {
entry:
  %0 = getelementptr inbounds %struct.st, %struct.st addrspace(1)* %object, i32 0, i32 0
  %1 = load %struct.inner, %struct.inner addrspace(1)* %0, align 4
  %2 = extractvalue %struct.inner %1, 0
  %3 = fadd float %2, 1.000000e+00
  %4 = insertvalue %struct.inner %1, float %3, 0
  store %struct.inner %4, %struct.inner addrspace(1)* %0
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!0}
!opencl.used.extensions = !{!1}
!opencl.used.optional.core.features = !{!1}
!opencl.compiler.options = !{!1}

!0 = !{i32 1, i32 2}
!1 = !{}
