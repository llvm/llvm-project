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
; CHECK: {{^[^: ]+}}: # %catch.pad
  %catch = catchpad [i32 5]
    to label %catch.body unwind label %catch.end
catch.body:
  %exn = call i8 addrspace(1)* @llvm.eh.exceptionpointer.p1i8(token %catch)
  %cast_exn = bitcast i8 addrspace(1)* %exn to i32 addrspace(1)*
  ; CHECK: movq %rax, %rcx
  ; CHECK-NEXT: callq g
  call void @g(i32 addrspace(1)* %cast_exn)
  catchret %catch to label %exit
catch.end:
  catchendpad unwind to caller
exit:
  ret void
}
