; REQUIRES: aarch64-registered-target
; RUN: opt -module-summary %s -o %t.bc
; RUN: llvm-lto2 run %t.bc -o %t.o -save-temps \
; RUN:   -r %t.bc,compute,px \
; RUN:   -mcpu=neoverse-v1 -O3 \
; RUN:   -vector-library=ArmPL
; RUN: llvm-nm %t.o.1 | FileCheck %s

; This test verifies that the VecLib propagation in LTO prevents a crash
; when compiling scalable vector frem operations.

; CHECK: compute

target triple = "aarch64-unknown-linux-gnu"

define fastcc <vscale x 2 x double> @compute(<vscale x 2 x double> %0) {
entry:
  %1 = frem <vscale x 2 x double> %0, zeroinitializer
  ret <vscale x 2 x double> %1
}
