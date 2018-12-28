; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-LLVM:      call spir_func void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 2, i32 3, i32 0)
; CHECK-LLVM-NEXT: call spir_func void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 2, i32 3, i32 1)
; CHECK-LLVM-NEXT: call spir_func void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 2, i32 3, i32 2)
; CHECK-LLVM-NEXT: call spir_func void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 2, i32 3, i32 3)

; CHECK-LLVM-NEXT: call spir_func void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 1, i32 3, i32 0)
; CHECK-LLVM-NEXT: call spir_func void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 1, i32 3, i32 1)
; CHECK-LLVM-NEXT: call spir_func void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 1, i32 3, i32 2)
; CHECK-LLVM-NEXT: call spir_func void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 1, i32 3, i32 3)

; CHECK-LLVM-NEXT: call spir_func void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 4, i32 3, i32 0)
; CHECK-LLVM-NEXT: call spir_func void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 4, i32 3, i32 1)
; CHECK-LLVM-NEXT: call spir_func void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 4, i32 3, i32 2)
; CHECK-LLVM-NEXT: call spir_func void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 4, i32 3, i32 3)

; global | acquire_release
; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[MemSema1:[0-9]+]] 516
; local | acquire_release
; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[MemSema2:[0-9]+]] 260
; image | acquire_release
; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[MemSema3:[0-9]+]] 2052

; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[ScopeWorkItem:[0-9]+]] 4
; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[ScopeWorkGroup:[0-9]+]] 2
; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[ScopeDevice:[0-9]+]] 1
; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[ScopeCrossDevice:[0-9]+]] 0

; CHECK-SPIRV: 3 MemoryBarrier [[ScopeWorkItem]] [[MemSema1]]
; CHECK-SPIRV-NEXT: 3 MemoryBarrier [[ScopeWorkGroup]] [[MemSema1]]
; CHECK-SPIRV-NEXT: 3 MemoryBarrier [[ScopeDevice]] [[MemSema1]]
; CHECK-SPIRV-NEXT: 3 MemoryBarrier [[ScopeCrossDevice]] [[MemSema1]]

; CHECK-SPIRV-NEXT: 3 MemoryBarrier [[ScopeWorkItem]] [[MemSema2]]
; CHECK-SPIRV-NEXT: 3 MemoryBarrier [[ScopeWorkGroup]] [[MemSema2]]
; CHECK-SPIRV-NEXT: 3 MemoryBarrier [[ScopeDevice]] [[MemSema2]]
; CHECK-SPIRV-NEXT: 3 MemoryBarrier [[ScopeCrossDevice]] [[MemSema2]]

; CHECK-SPIRV-NEXT: 3 MemoryBarrier [[ScopeWorkItem]] [[MemSema3]]
; CHECK-SPIRV-NEXT: 3 MemoryBarrier [[ScopeWorkGroup]] [[MemSema3]]
; CHECK-SPIRV-NEXT: 3 MemoryBarrier [[ScopeDevice]] [[MemSema3]]
; CHECK-SPIRV-NEXT: 3 MemoryBarrier [[ScopeCrossDevice]] [[MemSema3]]

; ModuleID = 'OpMemoryBarrier.ll'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @test() #0 !kernel_arg_addr_space !0 !kernel_arg_access_qual !0 !kernel_arg_type !0 !kernel_arg_base_type !0 !kernel_arg_type_qual !0 {
entry:
  call spir_func void @_Z22atomic_work_item_fencejii(i32 2, i32 3, i32 0) ; global mem fence + memory_scope_work_item
  call spir_func void @_Z22atomic_work_item_fencejii(i32 2, i32 3, i32 1) ; global mem fence + memory_scope_work_group
  call spir_func void @_Z22atomic_work_item_fencejii(i32 2, i32 3, i32 2) ; global mem fence + memory_scope_device
  call spir_func void @_Z22atomic_work_item_fencejii(i32 2, i32 3, i32 3) ; global mem fence + memory_scope_all_svm_devices

  call spir_func void @_Z22atomic_work_item_fencejii(i32 1, i32 3, i32 0) ; local mem fence + memory_scope_work_item
  call spir_func void @_Z22atomic_work_item_fencejii(i32 1, i32 3, i32 1) ; local mem fence + memory_scope_work_group
  call spir_func void @_Z22atomic_work_item_fencejii(i32 1, i32 3, i32 2) ; local mem fence + memory_scope__devices
  call spir_func void @_Z22atomic_work_item_fencejii(i32 1, i32 3, i32 3) ; local mem fence + memory_scope_all_svm_devices

  call spir_func void @_Z22atomic_work_item_fencejii(i32 4, i32 3, i32 0) ; image mem fence + memory_scope_work_item
  call spir_func void @_Z22atomic_work_item_fencejii(i32 4, i32 3, i32 1) ; image mem fence + memory_scope_work_group
  call spir_func void @_Z22atomic_work_item_fencejii(i32 4, i32 3, i32 2) ; image mem fence + memory_scope__devices
  call spir_func void @_Z22atomic_work_item_fencejii(i32 4, i32 3, i32 3) ; image mem fence + memory_scope_all_svm_devices
  ret void
}

declare spir_func void @_Z22atomic_work_item_fencejii(i32, i32, i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!1}
!opencl.ocl.version = !{!2}
!opencl.used.extensions = !{!0}
!opencl.used.optional.core.features = !{!0}
!opencl.compiler.options = !{!0}

!0 = !{}
!1 = !{i32 1, i32 2}
!2 = !{i32 2, i32 0}
