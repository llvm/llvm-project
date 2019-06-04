; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t
; RUN: FileCheck < %t %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

; CHECK-DAG: Capability Int8
; CHECK-DAG: Capability Int16
; CHECK-DAG: Capability Int64

; CHECK-DAG: TypeInt {{[0-9]+}} 8 0
; CHECK-DAG: TypeInt {{[0-9]+}} 16 0
; CHECK-DAG: TypeInt {{[0-9]+}} 64 0

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

@a = addrspace(1) global i8 0, align 1
@b = addrspace(1) global i16 0, align 2
@c = addrspace(1) global i64 0, align 8

!opencl.enable.FP_CONTRACT = !{}
!opencl.ocl.version = !{!0}
!opencl.spir.version = !{!0}
!opencl.used.extensions = !{!1}
!opencl.used.optional.core.features = !{!1}
!opencl.compiler.options = !{!1}

!0 = !{i32 2, i32 0}
!1 = !{}
