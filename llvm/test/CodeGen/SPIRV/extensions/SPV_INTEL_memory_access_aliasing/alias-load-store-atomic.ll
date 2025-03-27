; Check aliasing information translation on atomic load and store

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs --spirv-ext=+SPV_INTEL_memory_access_aliasing %s -o - | FileCheck %s

; CHECK: OpCapability MemoryAccessAliasingINTEL
; CHECK: OpExtension "SPV_INTEL_memory_access_aliasing"
; CHECK: %[[#Domain1:]] = OpAliasDomainDeclINTEL
; CHECK: %[[#Scope1:]] = OpAliasScopeDeclINTEL %[[#Domain1]]
; CHECK: %[[#List1:]] = OpAliasScopeListDeclINTEL %[[#Scope1]]
; CHECK: OpDecorate %[[#Load:]] NoAliasINTEL %[[#List1]]
; CHECK: %[[#Load:]] = OpAtomicLoad

define spir_func i32 @test_load(ptr addrspace(4) %object) #0 {
entry:
  %0 = call spir_func i32 @_Z18__spirv_AtomicLoadPU3AS4iii(ptr addrspace(4) %object, i32 1, i32 16), !noalias !1
  ret i32 %0
}

declare spir_func i32 @_Z18__spirv_AtomicLoadPU3AS4iii(ptr addrspace(4), i32, i32)

define spir_func void @test_store(ptr addrspace(4) %object, ptr addrspace(4) %expected, i32 %desired) {
entry:
  call spir_func void @_Z19__spirv_AtomicStorePU3AS4iiii(ptr addrspace(4) %object, i32 1, i32 16, i32 %desired), !noalias !4
  ret void
}

declare spir_func void @_Z19__spirv_AtomicStorePU3AS4iiii(ptr addrspace(4), i32, i32, i32)

!1 = !{!2}
!2 = distinct !{!2, !3}
!3 = distinct !{!3}
!4 = !{!5}
!5 = distinct !{!5, !6}
!6 = distinct !{!6}
