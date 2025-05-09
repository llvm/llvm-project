; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

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
; CHECK-SPIRV-NEXT:   %[[#desired:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:        OpAtomicStore %[[#object]] %[[#]] %[[#]] %[[#desired]]
; CHECK-SPIRV-LABEL:  OpFunctionEnd

define spir_func void @test_store(i32 addrspace(4)* %object, i32 %desired) {
entry:
  call spir_func void @_Z12atomic_storePVU3AS4U7_Atomicii(i32 addrspace(4)* %object, i32 %desired)
  ret void
}

declare spir_func i32 @_Z11atomic_loadPVU3AS4U7_Atomici(i32 addrspace(4)*)
declare spir_func void @_Z12atomic_storePVU3AS4U7_Atomicii(i32 addrspace(4)*, i32)

; The goal of @test_typesX() cases is to ensure that a correct pointer type
; is deduced from the Value argument of OpAtomicLoad/OpAtomicStore. There is
; no need to add more pattern matching rules to be sure that the pointer type
; is valid, it's enough that `spirv-val` considers the output valid as it
; checks the same condition while validating the output.

define spir_func void @test_types1(ptr addrspace(1) %ptr, float %val) {
entry:
  %r = call spir_func float @atomic_load(ptr addrspace(1) %ptr)
  ret void
}

define spir_func void @test_types2(ptr addrspace(1) %ptr, float %val) {
entry:
  call spir_func void @atomic_store(ptr addrspace(1) %ptr, float %val)
  ret void
}

define spir_func void @test_types3(i64 noundef %arg, float %val) {
entry:
  %ptr1 = inttoptr i64 %arg to float addrspace(1)*
  %r = call spir_func float @atomic_load(ptr addrspace(1) %ptr1)
  ret void
}

define spir_func void @test_types4(i64 noundef %arg, float %val) {
entry:
  %ptr2 = inttoptr i64 %arg to float addrspace(1)*
  call spir_func void @atomic_store(ptr addrspace(1) %ptr2, float %val)
  ret void
}

declare spir_func float @atomic_load(ptr addrspace(1))
declare spir_func void @atomic_store(ptr addrspace(1), float)
