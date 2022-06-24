; This fails because the linker renames the non-opaque type not the opaque
; one...

; RUN: echo "%%Ty = type opaque @GV = external global %%Ty*" | llvm-as > %t.1.bc
; RUN: llvm-as --opaque-pointers=0 < %s > %t.2.bc
; RUN: llvm-link --opaque-pointers=0 %t.1.bc %t.2.bc -S | FileCheck %s
; CHECK: = global %Ty

%Ty = type {i32}

@GV = global %Ty* null
