; RUN: opt < %s -mtriple=riscv32-unknown-cheriotrtos -passes=inferattrs -S | FileCheck %s --check-prefix=CHERIOT
; RUN: opt < %s -mtriple=riscv32-unknown-unknown -passes=inferattrs -S | FileCheck %s --check-prefix=NOCHERIOT

; heap_allocate / heap_free only get allocation attributes on CHERIoT RTOS.

target datalayout = "e-m:e-p:32:32-pe200:64:64:64:32-i64:64-n32-S128-A200-P200-G200"

; void *heap_allocate(Timeout *timeout, AllocatorCapability heapCapability,
;                     size_t size, uint32_t flags);
; CHERIOT: declare noalias ptr addrspace(200) @_Z13heap_allocateP7TimeoutU19__sealed_capabilityP24AllocatorCapabilityStatejj(ptr addrspace(200), ptr addrspace(200), i32, i32) addrspace(200) [[HEAP_ALLOCATE:#[0-9]+]]
; NOCHERIOT: declare ptr addrspace(200) @_Z13heap_allocateP7TimeoutU19__sealed_capabilityP24AllocatorCapabilityStatejj(ptr addrspace(200), ptr addrspace(200), i32, i32) addrspace(200){{$}}
declare ptr addrspace(200) @_Z13heap_allocateP7TimeoutU19__sealed_capabilityP24AllocatorCapabilityStatejj(ptr addrspace(200), ptr addrspace(200), i32, i32) addrspace(200)

; int heap_free(AllocatorCapability heapCapability, void *ptr);
; CHERIOT: declare i32 @_Z9heap_freeU19__sealed_capabilityP24AllocatorCapabilityStatePv(ptr addrspace(200), ptr addrspace(200) allocptr) addrspace(200) [[HEAP_FREE:#[0-9]+]]
; NOCHERIOT: declare i32 @_Z9heap_freeU19__sealed_capabilityP24AllocatorCapabilityStatePv(ptr addrspace(200), ptr addrspace(200)) addrspace(200){{$}}
declare i32 @_Z9heap_freeU19__sealed_capabilityP24AllocatorCapabilityStatePv(ptr addrspace(200), ptr addrspace(200)) addrspace(200)

; CHERIOT-DAG: attributes [[HEAP_ALLOCATE]] = { nofree allockind("alloc,zeroed") allocsize(2) "alloc-family"="cheriot_heap_allocate" }
; CHERIOT-DAG: attributes [[HEAP_FREE]] = { allockind("free") "alloc-family"="cheriot_heap_allocate" }
