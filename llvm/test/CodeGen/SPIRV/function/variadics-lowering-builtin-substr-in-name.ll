; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64 < %s 2>&1 | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64 < %s -o - -filetype=obj | spirv-val %}

; Variadics lowering will introduce a buffer that will be accessed through GEP.
; We don't lower SPIR-V builtins or printf, but make sure these functions aren't
; identified as such and are lowered.

; CHECK-COUNT-6: OpInBoundsPtrAccessChain

@.str = private unnamed_addr constant [4 x i8] c"hey\00", align 1

define dso_local noundef i32 @main() {
  call void (ptr, ...) @_Z16my__spirv_printfPKcz(ptr noundef @.str, i32 noundef 5, i32 noundef 6)
  call void (ptr, ...) @_Z16your__spirv_funcPKcz(ptr noundef @.str, i32 noundef 6, i32 noundef 7)
  call void (ptr, ...) @_Z22their_cool_printf_funcPKcz(ptr noundef @.str, i32 noundef 8, i32 noundef 9)
  ret i32 0
}

declare void @_Z16my__spirv_printfPKcz(ptr noundef, ...)

declare void @_Z16your__spirv_funcPKcz(ptr noundef, ...)

declare void @_Z22their_cool_printf_funcPKcz(ptr noundef, ...)
