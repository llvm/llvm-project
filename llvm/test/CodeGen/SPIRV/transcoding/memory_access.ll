; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-NOT: OpStore %[[#]] %[[#]] Volatile Aligned 8
; CHECK-SPIRV:     OpStore %[[#]] %[[#]] Volatile|Aligned 8
; CHECK-SPIRV-NOT: %[[#]] = OpLoad %[[#]] %[[#]] Volatile Aligned 8
; CHECK-SPIRV:     %[[#]] = OpLoad %[[#]] %[[#]] Volatile|Aligned 8
; CHECK-SPIRV:     %[[#]] = OpLoad %[[#]] %[[#]] Aligned 4
; CHECK-SPIRV-NOT: %[[#]] = OpLoad %[[#]] %[[#]] Volatile Aligned 8
; CHECK-SPIRV:     %[[#]] = OpLoad %[[#]] %[[#]] Volatile|Aligned 8
; CHECK-SPIRV-NOT: %[[#]] = OpLoad %[[#]] %[[#]] Volatile Aligned 0
; CHECK-SPIRV:     %[[#]] = OpLoad %[[#]] %[[#]] Volatile|Aligned 8
; CHECK-SPIRV-NOT: %[[#]] = OpLoad %[[#]] %[[#]] Volatile Aligned 8
; CHECK-SPIRV:     %[[#]] = OpLoad %[[#]] %[[#]] Volatile|Aligned|Nontemporal 8
; CHECK-SPIRV-NOT: OpStore %[[#]] %[[#]] Aligned 4
; CHECK-SPIRV:     OpStore %[[#]] %[[#]] Aligned|Nontemporal 4
; CHECK-SPIRV-NOT: OpStore %[[#]] %[[#]] Aligned 0
; CHECK-SPIRV:     OpStore %[[#]] %[[#]]

define spir_kernel void @test_load_store(i32 addrspace(1)* %destMemory, i32 addrspace(1)* %oldValues, i32 %newValue) {
entry:
  %ptr = alloca i32 addrspace(4)*, align 8
  %0 = addrspacecast i32 addrspace(1)* %oldValues to i32 addrspace(4)*
  store volatile i32 addrspace(4)* %0, i32 addrspace(4)** %ptr, align 8
  %1 = load volatile i32 addrspace(4)*, i32 addrspace(4)** %ptr, align 8
  %2 = load i32, i32 addrspace(4)* %1, align 4
  %call = call spir_func i32 @_Z14atomic_cmpxchgPVU3AS1iii(i32 addrspace(1)* %destMemory, i32 %2, i32 %newValue)
  %3 = load volatile i32 addrspace(4)*, i32 addrspace(4)** %ptr, align 8
  %4 = load volatile i32 addrspace(4)*, i32 addrspace(4)** %ptr
  %5 = load volatile i32 addrspace(4)*, i32 addrspace(4)** %ptr, align 8, !nontemporal !9
  %arrayidx = getelementptr inbounds i32, i32 addrspace(4)* %3, i64 0
  store i32 %call, i32 addrspace(4)* %arrayidx, align 4, !nontemporal !9
  store i32 addrspace(4)* %5, i32 addrspace(4)** %ptr
  ret void
}

declare spir_func i32 @_Z14atomic_cmpxchgPVU3AS1iii(i32 addrspace(1)*, i32, i32)

!9 = !{i32 1}
