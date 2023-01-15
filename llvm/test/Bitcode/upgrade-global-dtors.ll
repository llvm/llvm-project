; RUN: llvm-dis < %s.bc | FileCheck %s
; RUN: verify-uselistorder < %s.bc

; The 2-field form @llvm.global_dtors will be upgraded when reading bitcode.
; CHECK: @llvm.global_dtors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr null, ptr null }, { i32, ptr, ptr } { i32 65534, ptr null, ptr null }]
