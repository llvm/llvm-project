; RUN: llvm-as -o %t0 %s
; RUN: cp %t0 %t1
; RUN: llvm-dis %t0 %t1
; RUN: FileCheck %s < %t0.ll
; RUN: FileCheck %s < %t1.ll

; Test that if we disassemble the same bitcode twice, the type names are
; unchanged between the two. This protects against a bug whereby state was
; preserved across inputs and the types ended up with different names.

; CHECK: %Foo = type { ptr }
%Foo = type { ptr }

; CHECK: @foo = global %Foo zeroinitializer
@foo = global %Foo zeroinitializer
