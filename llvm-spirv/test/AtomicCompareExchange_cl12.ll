; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t
; RUN: FileCheck < %t %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK: 3 Source 3 102000

; CHECK: Name [[Pointer:[0-9]+]] "object"
; CHECK: Name [[Comparator:[0-9]+]] "expected"
; CHECK: Name [[Value:[0-9]+]] "desired"
; CHECK: 4 TypeInt [[int:[0-9]+]] 32 0
; CHECK: Constant [[int]] [[DeviceScope:[0-9]+]] 1
; CHECK: Constant [[int]] [[SequentiallyConsistent_MS:[0-9]+]] 16
; CHECK: 4 TypePointer [[int_ptr:[0-9]+]] 5 [[int]]

; Function Attrs: nounwind
define spir_func i32 @test(i32 addrspace(1)* %object, i32 %expected, i32 %desired) #0 {
; CHECK: FunctionParameter [[int_ptr]] [[Pointer]]
; CHECK: FunctionParameter [[int]] [[Comparator]]
; CHECK: FunctionParameter [[int]] [[Value]]
entry:
  %object.addr = alloca i32 addrspace(1)*, align 4
  %expected.addr = alloca i32, align 4
  %desired.addr = alloca i32, align 4
  %res = alloca i32, align 4
  store i32 addrspace(1)* %object, i32 addrspace(1)** %object.addr, align 4
  store i32 %expected, i32* %expected.addr, align 4
  store i32 %desired, i32* %desired.addr, align 4
  %0 = load i32 addrspace(1)*, i32 addrspace(1)** %object.addr, align 4
  %1 = load i32, i32* %expected.addr, align 4
  %2 = load i32, i32* %desired.addr, align 4

  %call = call spir_func i32 @_Z14atomic_cmpxchgPVU3AS1iii(i32 addrspace(1)* %0, i32 %1, i32 %2)
; CHECK 9 AtomicCompareExchange [[int]] [[result:[0-9]+]] [[Pointer]] [[DeviceScope]] [[SequentiallyConsistent_MS]] [[SequentiallyConsistent_MS]] [[Value]] [[Comparator]]

  store i32 %call, i32* %res, align 4
  %3 = load i32, i32* %res, align 4
  ret i32 %3
; CHECK 2 ReturnValue [[result]]
}

declare spir_func i32 @_Z14atomic_cmpxchgPVU3AS1iii(i32 addrspace(1)*, i32, i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!0}
!opencl.used.extensions = !{!1}
!opencl.used.optional.core.features = !{!1}
!opencl.compiler.options = !{!1}

!0 = !{i32 1, i32 2}
!1 = !{}
