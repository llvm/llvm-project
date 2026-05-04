; RUN: llvm-as %s -o %t.bc
; RUN: llvm-lto -exported-symbol=compute -exported-symbol=armpl_vsinq_f64 -o %t.o %t.bc
; RUN: llvm-nm %t.o | FileCheck %s

; This test ensures that ArmPL vector library functions are preserved through
; the old LTO API (llvm-lto). TargetLibraryInfoImpl in LTO backend passes 
; receive the VecLib parameter from TargetMachine Options.

; CHECK: armpl_vsinq_f64
; CHECK: compute

target triple = "aarch64-unknown-linux-gnu"

@llvm.compiler.used = appending global [1 x ptr] [ptr @armpl_vsinq_f64], section "llvm.metadata"

declare aarch64_vector_pcs <2 x double> @armpl_vsinq_f64(<2 x double>)

define void @compute(ptr %out, ptr %in) {
entry:
  %v = load <2 x double>, ptr %in, align 16
  %result = call aarch64_vector_pcs <2 x double> @armpl_vsinq_f64(<2 x double> %v)
  store <2 x double> %result, ptr %out, align 16
  ret void
}
