; RUN: llc -mtriple=x86_64-pc-windows-coreclr < %s | FileCheck %s

declare void @ProcessCLRException()
declare i8 addrspace(1)* @llvm.eh.exceptionpointer.p1i8(token)
declare void @f()
declare void @g(i32 addrspace(1)*)

; CHECK-LABEL: test1: # @test1
define void @test1() personality i8* bitcast (void ()* @ProcessCLRException to i8*) {
entry:
  invoke void @f()
    to label %exit unwind label %catch.pad
catch.pad:
  %cs1 = catchswitch within none [label %catch.body] unwind to caller
catch.body:
  ; CHECK: {{^[^: ]+}}: # %catch.body
  ; CHECK: movq %rdx, %rcx
  ; CHECK-NEXT: callq g
  %catch = catchpad within %cs1 [i32 5]
  %exn = call i8 addrspace(1)* @llvm.eh.exceptionpointer.p1i8(token %catch)
  %cast_exn = bitcast i8 addrspace(1)* %exn to i32 addrspace(1)*
  call void @g(i32 addrspace(1)* %cast_exn)
  catchret from %catch to label %exit
exit:
  ret void
}
