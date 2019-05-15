; RUN: llvm-dis < %s.bc | FileCheck %s
; RUN: verify-uselistorder < %s.bc

; The 2-field form @llvm.global_dtors will be upgraded when reading bitcode.
; CHECK: @llvm.global_dtors = appending global [2 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* null, i8* null }, { i32, void ()*, i8* } { i32 65534, void ()* null, i8* null }]
