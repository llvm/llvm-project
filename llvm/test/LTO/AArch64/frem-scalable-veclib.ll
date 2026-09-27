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

define <vscale x 2 x double> @compute(<vscale x 2 x double> %a, <vscale x 2 x double> %b) {
entry:
  %rem = frem <vscale x 2 x double> %a, %b
  ret <vscale x 2 x double> %rem
}
