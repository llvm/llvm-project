; Verify that we accept calls to variables in the program AS:
; RUN: llvm-as -data-layout "P40" %s -o - | llvm-dis - | FileCheck %s
; CHECK: target datalayout = "P40"

; We should get a sensible error for a non-program address call:
; RUN: not llvm-as -data-layout "P39" %s -o /dev/null 2>&1 | FileCheck %s -check-prefix ERR-AS39
; ERR-AS39: error: '%0' defined with type 'ptr addrspace(40)' but expected 'ptr addrspace(39)'

; And also if we don't set a custom program address space:
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s -check-prefix ERR-AS0
; ERR-AS0: error: '%0' defined with type 'ptr addrspace(40)' but expected 'ptr'

%fun1 = type i16 (i16)
%funptr1 = type ptr addrspace(40)

@fun_ptr = global %funptr1 @fun

define i16 @fun(i16 %arg) addrspace(40) {
entry:
  %0 = load %funptr1, ptr @fun_ptr
  %result = call i16 %0(i16 %arg)
  ret i16 %result
}
