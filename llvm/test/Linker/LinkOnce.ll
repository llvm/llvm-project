; This fails because the linker renames the non-opaque type not the opaque 
; one...

; RUN: echo "@X = linkonce global i32 8" | llvm-as > %t.2.bc
; RUN: llvm-as < %s > %t.1.bc
; RUN: llvm-link %t.1.bc %t.2.bc -S

@X = linkonce global i32 7		; <i32*> [#uses=0]
