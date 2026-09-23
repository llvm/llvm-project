; RUN: opt -module-summary %s -o %t.bc
; RUN: llvm-lto2 run -save-temps -o %t.o %t.bc \
; RUN:   -r=%t.bc,compute,px \
; RUN:   -r=%t.bc,armpl_vsinq_f64
; RUN: llvm-dis %t.o.1.5.precodegen.bc -o - | FileCheck %s

; This test ensures that ArmPL vector library functions are preserved through
; the new LTO API (llvm-lto2). TargetLibraryInfoImpl in LTO backend passes 
; receive the VecLib parameter from TargetMachine Options.

; CHECK: @llvm.compiler.used = appending global [1 x ptr] [ptr @armpl_vsinq_f64]
; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vsinq_f64(<2 x double>)

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
