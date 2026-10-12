; This test checks that the single-file `runtime_bitcode` key is accepted as an
; alias for a one-element `runtime_bitcodes` list, and that `runtime_bitcodes`
; takes precedence when both keys are present.
; RUN: rm -rf %t && mkdir -p %t && cd %t
; RUN: llvm-as %S/runtimes/runtime_bitcodes_state_rt.ll -o runtime_bitcodes_state_rt.bc
; RUN: llvm-as %S/runtimes/runtime_bitcodes_callbacks_rt.ll -o runtime_bitcodes_callbacks_rt.bc
; RUN: opt < %s -passes=instrumentor -instrumentor-read-config-files=%S/runtime_bitcode_alias_config.json -S 2>&1 \
; RUN:   | FileCheck %s --check-prefix=ALIAS
; RUN: opt < %s -passes=instrumentor -instrumentor-read-config-files=%S/runtime_bitcode_alias_conflict_config.json -S 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CONFLICT

; ALIAS-NOT: warning
; ALIAS: @runtime_state = protected global i32 0, align 4

; CONFLICT: warning: configuration key 'runtime_bitcode' ignored in favor of 'runtime_bitcodes'
; CONFLICT: @runtime_state = protected global i32 0, align 4
; CONFLICT-NOT: @__runtime_bitcodes_pre_numeric

define i32 @test(i32 %lhs, i32 %rhs) {
entry:
  %result = add i32 %lhs, %rhs
  ret i32 %result
}
