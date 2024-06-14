; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s -check-prefix=AMD64
; RUN: llc < %s -mtriple=x86_64-win32 | FileCheck %s -check-prefix=WIN64
; PR8743
; TAILJMPri64 should not receive "callee-saved" registers beyond epilogue.

; AMD64: jmpq
; AMD64-NOT: %{{e[a-z]|rbx|rbp|r10|r12|r13|r14|r15}}

; WIN64: jmpq
; WIN64-NOT: %{{e[a-z]|rbx|rsi|rdi|rbp|r12|r13|r14|r15}}

%class = type { [8 x i8] }
%vt = type { ptr }

define ptr @_ZN4llvm9UnsetInit20convertInitializerToEPNS_5RecTyE(ptr
%this, ptr %Ty) align 2 {
entry:
  %vtable = load ptr, ptr %Ty, align 8
  %vfn = getelementptr inbounds ptr, ptr %vtable, i64 4
  %0 = load ptr, ptr %vfn, align 8
  %call = tail call ptr %0(ptr %Ty, ptr %this)
  ret ptr %call
}
