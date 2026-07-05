; RUN: not llc -filetype=obj -mtriple powerpc-ibm-aix-xcoff -o %t.o < %s 2>&1 | FileCheck %s
; RUN: not llc -filetype=asm -mtriple powerpc-ibm-aix-xcoff -o %t.o < %s 2>&1 | FileCheck %s
; RUN: not llc -filetype=obj -mtriple powerpc64-ibm-aix-xcoff -o %t.o < %s 2>&1 | FileCheck %s
; RUN: not llc -filetype=asm -mtriple powerpc64-ibm-aix-xcoff -o %t.o < %s 2>&1 | FileCheck %s
@x= common global i32 0, align 4

@y= alias i32, ptr @x

; Function Attrs: noinline nounwind optnone
define ptr @g() #0 {
entry:
  ret ptr @y
}
; CHECK: LLVM ERROR: Aliases to common variables are not allowed on AIX:
; CHECK-NEXT:        Alias attribute for y is invalid because x is common.
