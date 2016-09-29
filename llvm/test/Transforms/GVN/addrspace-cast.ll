; RUN: opt < %s -gvn -S | FileCheck %s
target datalayout = "e-m:e-p:16:16-p1:32:16-i32:16-i64:16-n8:16"

; In cases where two address spaces do not have the same size pointer, the
; input for the addrspacecast should not be used as a substitute for itself
; when manipulating the pointer.

; Check that we don't hit the assert in this scenario
define i8 @test(i32 %V, i32* %P) {
; CHECK-LABEL: @test(
; CHECK: load
  %P1 = getelementptr inbounds i32, i32* %P, i16 16

  store i32 %V, i32* %P1

  %P2 = addrspacecast i32* %P1 to i8 addrspace(1)*
  %P3 = getelementptr i8, i8 addrspace(1)* %P2, i32 2

  %A = load i8, i8 addrspace(1)* %P3
  ret i8 %A
}
