; RUN: opt < %s -passes=instcombine -S | FileCheck %s

; This test makes sure that we do not assume globals in address spaces other
; than 0 are able to be null.

@as0 = external global i32
@as1 = external addrspace(1) global i32

declare void @addrspace0(i32*)
declare void @addrspace1(i32 addrspace(1)*)

; CHECK: call void @addrspace0(i32* nonnull @as0)
; CHECK: call void @addrspace1(i32 addrspace(1)* @as1)

define void @test() {
  call void @addrspace0(i32* @as0)
  call void @addrspace1(i32 addrspace(1)* @as1)
  ret void
}
