; RUN: llc -mtriple=x86_64-pc-linux -mcpu=corei7 < %s | FileCheck -check-prefix=CORE_LP64 %s
; RUN: llc -mtriple=x86_64-pc-linux -mcpu=atom < %s | FileCheck -check-prefix=ATOM_LP64 %s
; RUN: llc -mtriple=x86_64-pc-linux-gnux32 -mcpu=corei7 < %s | FileCheck -check-prefix=CORE_ILP32 %s
; RUN: llc -mtriple=x86_64-pc-linux-gnux32 -mcpu=atom < %s | FileCheck -check-prefix=ATOM_ILP32 %s

define i32 @bar(i32 %a) nounwind {
entry:
  %arr = alloca [400 x i32], align 16

; There is a 2x2 variation matrix here:
; Atoms use LEA to update the SP. Opcode bitness depends on data model.
; Cores use sub/add to update the SP. Opcode bitness depends on data model.

; CORE_LP64: subq $1608
; CORE_ILP32: subl $1608
; ATOM_LP64: leaq -1608
; ATOM_ILP32: leal -1608

  %call = call i32 @foo(i32 %a, ptr %arr) nounwind
  ret i32 %call

; CORE_LP64: addq $1608
; CORE_ILP32: addl $1608
; ATOM_LP64: leaq 1608
; ATOM_ILP32: leal 1608

}

declare i32 @foo(i32, ptr)

