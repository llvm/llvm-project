; ModuleID = 'AtomicCompareExchange_cl12.cl'
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s

; Check conversion of atomic_cmpxchng to atomic_compare_exchange_strong.

; Function Attrs: nounwind

; CHECK:         define spir_func i32 @test
; CHECK-LABEL:   entry
; CHECK:         [[PTR:%expected[0-9]*]] = alloca i32, align 4
; CHECK:         store i32 {{.*}}, i32* [[PTR]]
; CHECK:         call spir_func i1 @_Z39atomic_compare_exchange_strong_explicit{{.*}}%object, i32* [[PTR]], i32 %desired, i32 5, i32 5, i32 2)
; CHECK-NEXT;         load i32* [[PTR]]
define spir_func i32 @test(i32 addrspace(1)* %object, i32 %expected, i32 %desired) #0 {
entry:
  %call = tail call spir_func i32 @_Z14atomic_cmpxchgPVU3AS1iii(i32 addrspace(1)* %object, i32 %expected, i32 %desired) #2
  ret i32 %call
}

declare spir_func i32 @_Z14atomic_cmpxchgPVU3AS1iii(i32 addrspace(1)*, i32, i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!0}
!opencl.used.extensions = !{!1}
!opencl.used.optional.core.features = !{!1}
!opencl.compiler.options = !{!1}

!0 = !{i32 1, i32 2}
!1 = !{}
