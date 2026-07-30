; RUN: opt -passes=instcombine -S < %s | FileCheck %s

; When a pointer is addrspacecasted to a another addr space, we cannot assume
; anything about the new bits.

target datalayout = "p:32:32-p3:32:32-p4:64:64"

; CHECK-LABEL: @test_shift
; CHECK-NOT: ret i64 0
define i64 @test_shift(ptr %p) {
  %g = addrspacecast ptr %p to ptr addrspace(4)
  %i = ptrtoint ptr addrspace(4) %g to i64
  %shift = lshr i64 %i, 32
  ret i64 %shift
}

; CHECK-LABEL: @test_null
; A null pointer casted to another addr space may no longer have null value.
; CHECK-NOT: ret i32 0
define i32 @test_null() {
  %g = addrspacecast ptr null to ptr addrspace(3)
  %i = ptrtoint ptr addrspace(3) %g to i32
  ret i32 %i
}
