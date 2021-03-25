; REQUIRES: powerpc-registered-target
; RUN: llvm-as < %s > %t1
; RUN: llvm-lto %t1 | FileCheck %s

target triple = "powerpc-ibm-aix"

define i32 @main() {
entry:
  ret i32 42
}
; CHECK: Wrote native object file

