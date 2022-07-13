;; TODO: The alias offset doesn't refer to any sub-element.
; RUN: not --crash llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -data-sections=false < %s 2>&1 | FileCheck --check-prefix=ERROR %s

; ERROR: Aliases with offset 1 were not emitted.

@ConstVector = global <2 x i64> <i64 1, i64 2>
@var = alias i64, getelementptr inbounds (i8, ptr @ConstVector, i32 1)
