; RUN: not --crash llc -global-isel -mtriple=powerpc-unknown-linux-gnu \
; RUN:   -o - < %s 2>&1 | FileCheck %s --check-prefix=BE
; RUN: not --crash llc -global-isel -mtriple=powerpcle-unknown-linux-gnu \
; RUN:   -o - < %s 2>&1 | FileCheck %s --check-prefix=BIT32

; BE: LLVM ERROR: unable to translate in big endian mode

; BIT32: LLVM ERROR: unable to legalize instruction: [[LOAD:%[0-9]+]]:_(s64) = G_LOAD{{.*}}load (s64) from constant-pool

define double @foo() {
  entry:
    ret double 1.000000e+00
}
