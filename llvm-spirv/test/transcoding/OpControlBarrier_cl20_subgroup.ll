; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-LLVM: call spir_func void @_Z17sub_group_barrierji(i32 2, i32 1) [[attr:#[0-9]+]]
; CHECK-LLVM-NEXT: call spir_func void @_Z17sub_group_barrierji(i32 1, i32 1) [[attr]]
; CHECK-LLVM-NEXT: call spir_func void @_Z17sub_group_barrierji(i32 4, i32 1) [[attr]]
; CHECK-LLVM-NEXT: call spir_func void @_Z17sub_group_barrierji(i32 3, i32 1) [[attr]]
; CHECK-LLVM-NEXT: call spir_func void @_Z17sub_group_barrierji(i32 5, i32 1) [[attr]]
; CHECK-LLVM-NEXT: call spir_func void @_Z17sub_group_barrierji(i32 7, i32 1) [[attr]]

; CHECK-LLVM-NEXT: call spir_func void @_Z17sub_group_barrierji(i32 2, i32 0) [[attr]]
; CHECK-LLVM-NEXT: call spir_func void @_Z17sub_group_barrierji(i32 2, i32 1) [[attr]]
; CHECK-LLVM-NEXT: call spir_func void @_Z17sub_group_barrierji(i32 2, i32 2) [[attr]]
; CHECK-LLVM-NEXT: call spir_func void @_Z17sub_group_barrierji(i32 2, i32 3) [[attr]]

; CHECK-LLVM-NEXT: call spir_func void @_Z17sub_group_barrierji(i32 1, i32 0) [[attr]]
; CHECK-LLVM-NEXT: call spir_func void @_Z17sub_group_barrierji(i32 1, i32 1) [[attr]]
; CHECK-LLVM-NEXT: call spir_func void @_Z17sub_group_barrierji(i32 1, i32 2) [[attr]]
; CHECK-LLVM-NEXT: call spir_func void @_Z17sub_group_barrierji(i32 1, i32 3) [[attr]]

; CHECK-LLVM-NEXT: call spir_func void @_Z17sub_group_barrierji(i32 4, i32 0) [[attr]]
; CHECK-LLVM-NEXT: call spir_func void @_Z17sub_group_barrierji(i32 4, i32 1) [[attr]]
; CHECK-LLVM-NEXT: call spir_func void @_Z17sub_group_barrierji(i32 4, i32 2) [[attr]]
; CHECK-LLVM-NEXT: call spir_func void @_Z17sub_group_barrierji(i32 4, i32 3) [[attr]]

; CHECK-LLVM: attributes [[attr]] = { noduplicate nounwind }

; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[MemSema1:[0-9]+]] 528
; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[MemSema2:[0-9]+]] 272
; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[MemSema3:[0-9]+]] 2064
; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[MemSema4:[0-9]+]] 784
; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[MemSema5:[0-9]+]] 2320
; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[MemSema6:[0-9]+]] 2832

; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[ScopeWorkItem:[0-9]+]] 4
; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[ScopeWorkGroup:[0-9]+]] 2
; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[ScopeDevice:[0-9]+]] 1
; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[ScopeCrossDevice:[0-9]+]] 0
; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[ScopeSubGroup:[0-9]+]] 3

; CHECK-SPIRV: 4 ControlBarrier [[ScopeSubGroup]] [[ScopeWorkGroup]] [[MemSema1]]
; CHECK-SPIRV-NEXT: 4 ControlBarrier [[ScopeSubGroup]] [[ScopeWorkGroup]] [[MemSema2]]
; CHECK-SPIRV-NEXT: 4 ControlBarrier [[ScopeSubGroup]] [[ScopeWorkGroup]] [[MemSema3]]
; CHECK-SPIRV-NEXT: 4 ControlBarrier [[ScopeSubGroup]] [[ScopeWorkGroup]] [[MemSema4]]
; CHECK-SPIRV-NEXT: 4 ControlBarrier [[ScopeSubGroup]] [[ScopeWorkGroup]] [[MemSema5]]
; CHECK-SPIRV-NEXT: 4 ControlBarrier [[ScopeSubGroup]] [[ScopeWorkGroup]] [[MemSema6]]

; CHECK-SPIRV-NEXT: 4 ControlBarrier [[ScopeSubGroup]] [[ScopeWorkItem]] [[MemSema1]]
; CHECK-SPIRV-NEXT: 4 ControlBarrier [[ScopeSubGroup]] [[ScopeWorkGroup]] [[MemSema1]]
; CHECK-SPIRV-NEXT: 4 ControlBarrier [[ScopeSubGroup]] [[ScopeDevice]] [[MemSema1]]
; CHECK-SPIRV-NEXT: 4 ControlBarrier [[ScopeSubGroup]] [[ScopeCrossDevice]] [[MemSema1]]

; CHECK-SPIRV-NEXT: 4 ControlBarrier [[ScopeSubGroup]] [[ScopeWorkItem]] [[MemSema2]]
; CHECK-SPIRV-NEXT: 4 ControlBarrier [[ScopeSubGroup]] [[ScopeWorkGroup]] [[MemSema2]]
; CHECK-SPIRV-NEXT: 4 ControlBarrier [[ScopeSubGroup]] [[ScopeDevice]] [[MemSema2]]
; CHECK-SPIRV-NEXT: 4 ControlBarrier [[ScopeSubGroup]] [[ScopeCrossDevice]] [[MemSema2]]

; CHECK-SPIRV-NEXT: 4 ControlBarrier [[ScopeSubGroup]] [[ScopeWorkItem]] [[MemSema3]]
; CHECK-SPIRV-NEXT: 4 ControlBarrier [[ScopeSubGroup]] [[ScopeWorkGroup]] [[MemSema3]]
; CHECK-SPIRV-NEXT: 4 ControlBarrier [[ScopeSubGroup]] [[ScopeDevice]] [[MemSema3]]
; CHECK-SPIRV-NEXT: 4 ControlBarrier [[ScopeSubGroup]] [[ScopeCrossDevice]] [[MemSema3]]

; ModuleID = 'sub_group_barrier.cl'
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @test() #0 {
entry:
  call spir_func void @_Z17sub_group_barrierj(i32 2) ; global mem fence
  call spir_func void @_Z17sub_group_barrierj(i32 1) ; local mem fence
  call spir_func void @_Z17sub_group_barrierj(i32 4) ; image mem fence

  call spir_func void @_Z17sub_group_barrierj(i32 3) ; global | local
  call spir_func void @_Z17sub_group_barrierj(i32 5) ; local | image
  call spir_func void @_Z17sub_group_barrierj(i32 7) ; global | local | image

  call spir_func void @_Z17sub_group_barrierji(i32 2, i32 0) ; global mem fence + memory_scope_work_item
  call spir_func void @_Z17sub_group_barrierji(i32 2, i32 1) ; global mem fence + memory_scope_work_group
  call spir_func void @_Z17sub_group_barrierji(i32 2, i32 2) ; global mem fence + memory_scope_device
  call spir_func void @_Z17sub_group_barrierji(i32 2, i32 3) ; global mem fence + memory_scope_all_svm_devices

  call spir_func void @_Z17sub_group_barrierji(i32 1, i32 0) ; local mem fence + memory_scope_work_item
  call spir_func void @_Z17sub_group_barrierji(i32 1, i32 1) ; local mem fence + memory_scope_work_group
  call spir_func void @_Z17sub_group_barrierji(i32 1, i32 2) ; local mem fence + memory_scope_device
  call spir_func void @_Z17sub_group_barrierji(i32 1, i32 3) ; local mem fence + memory_scope_all_svm_devices

  call spir_func void @_Z17sub_group_barrierji(i32 4, i32 0) ; image mem fence + memory_scope_work_item
  call spir_func void @_Z17sub_group_barrierji(i32 4, i32 1) ; image mem fence + memory_scope_work_group
  call spir_func void @_Z17sub_group_barrierji(i32 4, i32 2) ; image mem fence + memory_scope_device
  call spir_func void @_Z17sub_group_barrierji(i32 4, i32 3) ; image mem fence + memory_scope_all_svm_devices

  ret void
}

declare spir_func void @_Z17sub_group_barrierj(i32) #1
declare spir_func void @_Z17sub_group_barrierji(i32, i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!8}
!opencl.compiler.options = !{!8}

!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 0}
!8 = !{}
