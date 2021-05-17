; RUN: opt -print-memderefs -analyze -S < %s -enable-new-pm=0 -use-dereferenceable-at-point-semantics=1 | FileCheck %s --check-prefixes=CHECK
; RUN: opt -passes=print-memderefs -S < %s -disable-output  -use-dereferenceable-at-point-semantics=1 2>&1 | FileCheck %s --check-prefixes=CHECK

target datalayout = "e-i32:32:64"

; For the abstract machine model (before RS4GC), gc managed objects
; conceptually live forever.  But there may be non-managed objects which are
; freed.
; CHECK-LABEL: 'abstract_model'
; CHECK: %gc_ptr
; CHECK-NOT: %other_ptr
; FIXME: Can infer the gc pointer case
define void @abstract_model(i32 addrspace(1)* dereferenceable(8) %gc_ptr,
                            i32* dereferenceable(8) %other_ptr)
    gc "statepoint-example" {
entry:
  call void @mayfree()
  load i32, i32 addrspace(1)* %gc_ptr
  load i32, i32* %other_ptr
  ret void
}

; Can free any object accessible in memory
declare void @mayfree()
