; This test checks that Instrumentor links multiple runtime bitcode modules.
; The callback runtime uses state from the preceding runtime module.
; RUN: llvm-as %S/runtimes/runtime_bitcodes_state_rt.ll -o runtime_bitcodes_state_rt.bc
; RUN: llvm-as %S/runtimes/runtime_bitcodes_callbacks_rt.ll -o runtime_bitcodes_callbacks_rt.bc
; RUN: opt < %s -passes=instrumentor -instrumentor-read-config-files=%S/runtime_bitcodes_config.json -S | FileCheck %s

; CHECK-DAG: @runtime_state = protected global i32 0, align 4
; CHECK-DAG: @runtime_private_state = internal global i32 0, align 4
; CHECK-DAG: define protected void @__runtime_bitcodes_pre_numeric(

@runtime_state = external global i32

define i32 @test(i32 %lhs, i32 %rhs) {
entry:
  %result = add i32 %lhs, %rhs
  ret i32 %result
}
