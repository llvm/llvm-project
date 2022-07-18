; RUN: opt -S -passes=verify < %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

define i32 @check_verify_undef_token() gc "statepoint-example" {

entry:
    ; CHECK: ret i32 0
    ret i32 0

unreach:
    ; CHECK: token undef
    %token_call = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token undef, i32 0, i32 0)
    ret i32 1
}

declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token, i32, i32)
