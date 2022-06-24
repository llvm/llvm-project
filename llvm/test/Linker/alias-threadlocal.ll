; RUN: llvm-link %s %p/Inputs/alias-threadlocal-defs.ll -S -o - | FileCheck %s

; PR46297
; Verify that linking GlobalAliases preserves the thread_local attribute

; CHECK: @tlsvar1 = thread_local global i32 0, align 4
; CHECK: @tlsvar2 = hidden thread_local alias i32, i32* @tlsvar1

@tlsvar2 = external thread_local global i32, align 4
