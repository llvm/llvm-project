; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s -check-prefix=AMD64
; RUN: llc < %s -mtriple=x86_64-win32 | FileCheck %s -check-prefix=WIN64
; PR8743
; TAILJMPri64 should not receive "callee-saved" registers beyond epilogue.

; AMD64: jmpq
; AMD64-NOT: %{{e[a-z]|rbx|rbp|r10|r12|r13|r14|r15}}

; WIN64: jmpq
; WIN64-NOT: %{{e[a-z]|rbx|rsi|rdi|rbp|r12|r13|r14|r15}}

%class = type { [8 x i8] }
%vt = type { i32 (...)** }

define %vt* @_ZN4llvm9UnsetInit20convertInitializerToEPNS_5RecTyE(%class*
%this, %vt* %Ty) align 2 {
entry:
  %0 = bitcast %vt* %Ty to %vt* (%vt*, %class*)***
  %vtable = load %vt* (%vt*, %class*)*** %0, align 8
  %vfn = getelementptr inbounds %vt* (%vt*, %class*)** %vtable, i64 4
  %1 = load %vt* (%vt*, %class*)** %vfn, align 8
  %call = tail call %vt* %1(%vt* %Ty, %class* %this)
  ret %vt* %call
}
