; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc-ibm-aix-xcoff < \
; RUN: %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc64-ibm-aix-xcoff < \
; RUN: %s | FileCheck %s

define dso_local i32 @main() {
entry:
  unreachable

unreachabl_bb:
  ret i32 0
}

; CHECK:       .main:
; CHECK-NEXT:  # %bb.0:                                # %entry
; CHECK-NEXT:    trap

