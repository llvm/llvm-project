; This fails because the linker renames the non-opaque type not the opaque 
; one...

; RUN: echo " define linkonce void @foo() { ret void } " | \
; RUN:   llvm-as -o %t.2.bc
; RUN: llvm-as %s -o %t.1.bc
; RUN: llvm-link %t.1.bc %t.2.bc -S | FileCheck %s
; CHECK: linkonce{{.*}}foo

declare void @foo()

