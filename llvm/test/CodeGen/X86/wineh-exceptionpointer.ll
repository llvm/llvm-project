; RUN: llc -mtriple=x86_64-pc-windows-coreclr < %s | FileCheck %s

declare void @ProcessCLRException()
declare ptr addrspace(1) @llvm.eh.exceptionpointer.p1(token)
declare void @f()
declare void @g(ptr addrspace(1))

; CHECK-LABEL: test1: # @test1
define void @test1() personality ptr @ProcessCLRException {
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
  %exn = call ptr addrspace(1) @llvm.eh.exceptionpointer.p1(token %catch)
  call void @g(ptr addrspace(1) %exn) [ "funclet"(token %catch) ]
  catchret from %catch to label %exit
exit:
  ret void
}
