; RUN: opt -S -passes=globalopt < %s | FileCheck %s

@_Z17in_custom_section = internal global i8 42, section "CUSTOM"
@in_custom_section = internal dllexport alias i8, i8* @_Z17in_custom_section

; CHECK: @in_custom_section = internal dllexport global i8 42, section "CUSTOM"

@llvm.used = appending global [1 x i8*] [i8* @in_custom_section], section "llvm.metadata"
