; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck %s --check-prefix=PPC64
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc64-ibm-aix-xcoff -mcpu=pwr7 | FileCheck %s --check-prefix=PPC64

define void @t1(i8* %x) nounwind {
entry:
; PPC64: t1
  br label %L0

L0:
  br label %L1

L1:
  indirectbr i8* %x, [ label %L0, label %L1 ]
; PPC64: mtctr 3
; PPC64: bctr
}
