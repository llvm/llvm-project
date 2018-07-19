; RUN: llvm-link %p/Inputs/drop-debug.bc -o %t 2>&1 | FileCheck %s

;; drop-debug.bc was created from "void f(void) {}" with clang 3.5 and
; -gline-tables-only, so it contains old debug info.

; CHECK: WARNING: ignoring debug info with an invalid version (1) in {{.*}}/Inputs/drop-debug.bc
