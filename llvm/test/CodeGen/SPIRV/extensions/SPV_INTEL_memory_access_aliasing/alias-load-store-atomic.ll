; Do not attach aliasing decorations to load/store atomics since the extension is inconsistent.
; We cannot attach decorations to stores since they have no id (while we can for loads).

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs --spirv-ext=+SPV_INTEL_memory_access_aliasing %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_memory_access_aliasing %s -o - -filetype=obj | spirv-val %}

; CHECK-NOT: OpCapability MemoryAccessAliasingINTEL
; CHECK-NOT: OpExtension "SPV_INTEL_memory_access_aliasing"
; CHECK-NOT: OpDecorateId
; CHECK: %[[#LoadFun:]] = OpAtomicLoad
; CHECK: OpAtomicStore
; CHECK: %[[#LoadInst:]] = OpAtomicLoad
; CHECK: OpAtomicStore

define spir_func i32 @test_load_call(ptr addrspace(4) %object) #0 {
entry:
  %0 = call spir_func i32 @_Z18__spirv_AtomicLoadPU3AS4iii(ptr addrspace(4) %object, i32 1, i32 16), !noalias !1
  ret i32 %0
}

define spir_func void @test_store_call(ptr addrspace(4) %object, i32 %desired) {
entry:
  call spir_func void @_Z19__spirv_AtomicStorePU3AS4iiii(ptr addrspace(4) %object, i32 1, i32 16, i32 %desired), !noalias !4
  ret void
}

define spir_func i32 @test_load_instr(ptr addrspace(4) %object) #0 {
entry:
  %0 = load atomic i32, ptr addrspace(4) %object syncscope("singlethread") acquire, align 4, !noalias !7
  ret i32 %0
}

define spir_func void @test_store_instr(ptr addrspace(4) %object, i32 %desired) {
entry:
  store atomic i32 %desired, ptr addrspace(4) %object syncscope("singlethread") release, align 4, !noalias !10
  ret void
}

declare spir_func i32 @_Z18__spirv_AtomicLoadPU3AS4iii(ptr addrspace(4), i32, i32)
declare spir_func void @_Z19__spirv_AtomicStorePU3AS4iiii(ptr addrspace(4), i32, i32, i32)

!1 = !{!2}
!2 = distinct !{!2, !3}
!3 = distinct !{!3}
!4 = !{!5}
!5 = distinct !{!5, !6}
!6 = distinct !{!6}
!7 = !{!8}
!8 = distinct !{!8, !9}
!9 = distinct !{!9}
!10 = !{!11}
!11 = distinct !{!11, !12}
!12 = distinct !{!12}
