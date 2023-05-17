; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
; RUN: llvm-as %s -data-layout=P200 -o - | llvm-dis - -o - | FileCheck %s -check-prefix PROGAS200


; Check that variables in a nonzero program address space 200 can be used in a invoke instruction

define i8 @test_invoke(ptr %fnptr0, ptr addrspace(200) %fnptr200) personality ptr addrspace(200) @__gxx_personality_v0 {
  %explicit_as_0 = invoke addrspace(0) i8 %fnptr0(i32 0) to label %ok unwind label %lpad
  %explicit_as_42 = invoke addrspace(200) i8 %fnptr200(i32 0) to label %ok unwind label %lpad
  ; The following is only okay if the program address space is 200:
  %no_as = invoke i8 %fnptr200(i32 0) to label %ok unwind label %lpad
  ; CHECK: invoke-nonzero-program-addrspace.ll:[[@LINE-1]]:22: error: '%fnptr200' defined with type 'ptr addrspace(200)' but expected 'ptr'
ok:
  ret i8 0
lpad:
    %exn = landingpad {ptr, i32}
            cleanup
    unreachable
}

declare i32 @__gxx_personality_v0(...)


; PROGAS200:  target datalayout = "P200"
; PROGAS200:  define i8 @test_invoke(ptr %fnptr0, ptr addrspace(200) %fnptr200) addrspace(200) personality ptr addrspace(200) @__gxx_personality_v0 {
; PROGAS200:    %explicit_as_0 = invoke addrspace(0) i8 %fnptr0(i32 0)
; PROGAS200:    %explicit_as_42 = invoke addrspace(200) i8 %fnptr200(i32 0)
; PROGAS200:    %no_as = invoke addrspace(200) i8 %fnptr200(i32 0)
; PROGAS200:    ret i8 0
; PROGAS200:  }
