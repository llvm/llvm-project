; OpenCL C source:
; #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
; #pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
;
; void foo (volatile atomic_long *object, long desired) {
;   atomic_fetch_xor(object, desired);
;}

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t
; RUN: FileCheck < %t %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

; CHECK: Capability Int64Atomics

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

; Function Attrs: nounwind
define spir_func void @foo(i64 addrspace(4)* %object, i64 %desired) #0 {
entry:
  %call = tail call spir_func i64 @_Z16atomic_fetch_xorPVU3AS4U7_Atomicll(i64 addrspace(4)* %object, i64 %desired) #2
  ret void
}

declare spir_func i64 @_Z16atomic_fetch_xorPVU3AS4U7_Atomicll(i64 addrspace(4)*, i64) #1

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
