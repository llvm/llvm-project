; This test checks that linking runtime bitcode is reported as a change, so that
; module analyses cached before Instrumentor are invalidated. Nothing in @test
; is instrumented, so the linked runtime is the only change to the module.
; RUN: rm -rf %t && mkdir -p %t && cd %t
; RUN: llvm-as %S/runtimes/runtime_bitcodes_state_rt.ll -o runtime_bitcodes_state_rt.bc
; RUN: llvm-as %S/runtimes/runtime_bitcodes_callbacks_rt.ll -o runtime_bitcodes_callbacks_rt.bc
; RUN: opt < %s -passes='require<callgraph>,instrumentor,print-callgraph' \
; RUN:   -instrumentor-read-config-files=%S/runtime_bitcodes_config.json \
; RUN:   -disable-output 2>&1 | FileCheck %s

; CHECK: Call graph node for function: '__runtime_bitcodes_pre_numeric'

define void @test() {
entry:
  ret void
}
