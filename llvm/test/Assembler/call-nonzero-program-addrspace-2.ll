; RUN: llvm-as %s -data-layout=P200 -o /dev/null
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; Check that numbered variables in a nonzero program address space 200 can be used in a call instruction

define i8 @test_unnamed(i8(i32)*, i8(i32) addrspace(200)*) {
  %first = call i8 %0(i32 0) ; this is fine
  %second = call i8 %1(i32 0) ; this is also fine if it's the program AS
  ; CHECK: call-nonzero-program-addrspace-2.ll:[[@LINE-1]]:21: error: '%1' defined with type 'i8 (i32) addrspace(200)*'
  ret i8 0
}
