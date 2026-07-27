; RUN: llvm-as %S/runtimes/runtime_export_symbols_rt.ll -o runtime_export_symbols_rt.bc
; RUN: opt < %s -passes=instrumentor -instrumentor-read-config-files=%S/runtime_export_symbols_config.json -S | FileCheck %s

; CHECK: @runtime_export = linkonce_odr global i32 0, comdat, align 4
; CHECK: @runtime_internal = internal global i32 0, align 4

@runtime_export = external global i32
@runtime_internal = external global i32

define i32 @test(i32 %lhs, i32 %rhs) {
entry:
  %result = add i32 %lhs, %rhs
  ret i32 %result
}