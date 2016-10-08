; RUN: opt -S -rewrite-statepoints-for-gc < %s | FileCheck %s
; Ensure statepoints copy (valid) attributes from callsites.

declare void @f(i8 addrspace(1)* %obj)

; copy over norecurse noimplicitfloat to statepoint call
define void @test1(i8 addrspace(1)* %arg) gc "statepoint-example" {
; CHECK-LABEL: test1(
; CHECK: call token (i64, i32, void (i8 addrspace(1)*)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidp1i8f(i64 2882400000, i32 0, void (i8 addrspace(1)*)* @f, i32 1, i32 0, i8 addrspace(1)* %arg, i32 0, i32 0, i8 addrspace(1)* %arg) #1

 call void @f(i8 addrspace(1)* %arg) #1
 ret void
}


attributes #1 = { norecurse noimplicitfloat }
