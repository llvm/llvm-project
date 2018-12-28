; ModuleID = ''
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

; Check 'LLVM ==> SPIR-V ==> LLVM' conversion of atomic_store.

; Function Attrs: nounwind

; CHECK-LLVM:         define spir_func void @test
; CHECK-LLVM-LABEL:   entry
; CHECK-LLVM:         call spir_func void @_Z21atomic_store_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(i32 addrspace(4)* %object, i32 %desired, i32 5, i32 2)

; CHECK-SPIRV-LABEL:  5 Function
; CHECK-SPIRV-NEXT:   FunctionParameter {{[0-9]+}} [[object:[0-9]+]]
; CHECK-SPIRV-NEXT:   FunctionParameter
; CHECK-SPIRV-NEXT:   FunctionParameter {{[0-9]+}} [[desired:[0-9]+]]
; CHECK-SPIRV:        AtomicStore [[object]] {{[0-9]+}} {{[0-9]+}} [[desired]]
; CHECK-SPIRV-LABEL:  1 FunctionEnd

; Function Attrs: nounwind
define spir_func void @test(i32 addrspace(4)* %object, i32 addrspace(4)* %expected, i32 %desired) #0 {
entry:
  call spir_func void @_Z12atomic_storePVU3AS4U7_Atomicii(i32 addrspace(4)* %object, i32 %desired) #2
  ret void
}

declare spir_func void @_Z12atomic_storePVU3AS4U7_Atomicii(i32 addrspace(4)*, i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}

!0 = !{i32 1, i32 2}
!1 = !{i32 2, i32 0}
!2 = !{}
