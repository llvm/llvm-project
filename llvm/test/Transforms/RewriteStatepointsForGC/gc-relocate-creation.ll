; RUN: opt < %s -rewrite-statepoints-for-gc -S | FileCheck %s
; RUN: opt < %s -passes=rewrite-statepoints-for-gc -S | FileCheck %s

; This test is to verify gc.relocate can handle pointer to vector of
; pointers (<2 x i32 addrspace(1)*> addrspace(1)* in this case).
; The old scheme to create a gc.relocate of <2 x i32 addrspace(1)*> addrspace(1)*
; type will fail because llvm does not support mangling vector of pointers.
; The new scheme will create all gc.relocate to i8 addrspace(1)* type and
; then bitcast to the correct type.

declare void @foo()

declare void @use(...) "gc-leaf-function"

define void @test1(<2 x i32 addrspace(1)*> addrspace(1)* %obj) gc "statepoint-example" {
entry:
; CHECK: %obj.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %statepoint_token, i32 7, i32 7)
; CHECK-NEXT:  %obj.relocated.casted = bitcast i8 addrspace(1)* %obj.relocated to <2 x i32 addrspace(1)*> addrspace(1)*

  call void @foo() [ "deopt"() ]
  call void (...) @use(<2 x i32 addrspace(1)*> addrspace(1)* %obj)
  ret void
}
