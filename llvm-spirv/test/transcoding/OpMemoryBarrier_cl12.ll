; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-LLVM:      call spir_func void @_Z9mem_fencej(i32 2)
; CHECK-LLVM-NEXT: call spir_func void @_Z9mem_fencej(i32 1)
; CHECK-LLVM-NEXT: call spir_func void @_Z9mem_fencej(i32 4)
; CHECK-LLVM-NEXT: call spir_func void @_Z9mem_fencej(i32 3)
; CHECK-LLVM-NEXT: call spir_func void @_Z9mem_fencej(i32 5)
; CHECK-LLVM-NEXT: call spir_func void @_Z9mem_fencej(i32 7)

; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[MemSema1:[0-9]+]] 512
; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[MemSema2:[0-9]+]] 256
; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[MemSema3:[0-9]+]] 2048
; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[MemSema4:[0-9]+]] 768
; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[MemSema5:[0-9]+]] 2304
; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[MemSema6:[0-9]+]] 2816
; CHECK-SPIRV-DAG: 4 Constant {{[0-9]+}} [[ScopeWorkGroup:[0-9]+]] 2

; CHECK-SPIRV: 3 MemoryBarrier [[ScopeWorkGroup]] [[MemSema1]]
; CHECK-SPIRV-NEXT: 3 MemoryBarrier [[ScopeWorkGroup]] [[MemSema2]]
; CHECK-SPIRV-NEXT: 3 MemoryBarrier [[ScopeWorkGroup]] [[MemSema3]]
; CHECK-SPIRV-NEXT: 3 MemoryBarrier [[ScopeWorkGroup]] [[MemSema4]]
; CHECK-SPIRV-NEXT: 3 MemoryBarrier [[ScopeWorkGroup]] [[MemSema5]]
; CHECK-SPIRV-NEXT: 3 MemoryBarrier [[ScopeWorkGroup]] [[MemSema6]]

; ModuleID = 'OpMemoryBarrier.ll'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @test() #0 !kernel_arg_addr_space !0 !kernel_arg_access_qual !0 !kernel_arg_type !0 !kernel_arg_base_type !0 !kernel_arg_type_qual !0 {
entry:
  call spir_func void @_Z9mem_fencej(i32 2) ; global mem fence
  call spir_func void @_Z9mem_fencej(i32 1) ; local mem fence
  call spir_func void @_Z9mem_fencej(i32 4) ; image mem fence

  call spir_func void @_Z9mem_fencej(i32 3) ; global | local
  call spir_func void @_Z9mem_fencej(i32 5) ; local | image
  call spir_func void @_Z9mem_fencej(i32 7) ; global | local | image
  ret void
}

declare spir_func void @_Z9mem_fencej(i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!1}
!opencl.ocl.version = !{!1}
!opencl.used.extensions = !{!0}
!opencl.used.optional.core.features = !{!0}
!opencl.compiler.options = !{!0}

!0 = !{}
!1 = !{i32 1, i32 2}
