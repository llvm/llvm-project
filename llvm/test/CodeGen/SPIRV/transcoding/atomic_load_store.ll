; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;; Check 'LLVM ==> SPIR-V' conversion of atomic_load and atomic_store.

; CHECK-SPIRV-LABEL:  OpFunction
; CHECK-SPIRV-NEXT:   %[[#object:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:        %[[#ret:]] = OpAtomicLoad %[[#]] %[[#object]] %[[#]] %[[#]]
; CHECK-SPIRV:        OpReturnValue %[[#ret]]
; CHECK-SPIRV-LABEL:  OpFunctionEnd

define spir_func i32 @test_load(i32 addrspace(4)* %object) {
entry:
  %0 = call spir_func i32 @_Z11atomic_loadPVU3AS4U7_Atomici(i32 addrspace(4)* %object)
  ret i32 %0
}

; CHECK-SPIRV-LABEL:  OpFunction
; CHECK-SPIRV-NEXT:   %[[#object:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT:   OpFunctionParameter
; CHECK-SPIRV-NEXT:   %[[#desired:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:        OpAtomicStore %[[#object]] %[[#]] %[[#]] %[[#desired]]
; CHECK-SPIRV-LABEL:  OpFunctionEnd

define spir_func void @test_store(i32 addrspace(4)* %object, i32 addrspace(4)* %expected, i32 %desired) {
entry:
  call spir_func void @_Z12atomic_storePVU3AS4U7_Atomicii(i32 addrspace(4)* %object, i32 %desired)
  ret void
}

declare spir_func i32 @_Z11atomic_loadPVU3AS4U7_Atomici(i32 addrspace(4)*)

declare spir_func void @_Z12atomic_storePVU3AS4U7_Atomicii(i32 addrspace(4)*, i32)
