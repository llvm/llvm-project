; RUN: llvm-as %s -data-layout=P200 -o /dev/null
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; Check that variables in a nonzero program address space 200 can be used in a call instruction

define i8 @test(i8(i32)* %fnptr0, i8(i32) addrspace(200)* %fnptr200) {
  %first = call i8 %fnptr0(i32 0) ; this is fine
  %second = call i8 %fnptr200(i32 0) ; this is also fine if it's the program AS
  ; CHECK: call-nonzero-program-addrspace.ll:[[@LINE-1]]:21: error: '%fnptr200' defined with type 'i8 (i32) addrspace(200)*'
  ret i8 0
}

declare i32 @__gxx_personality_v0(...)
