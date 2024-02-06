; RUN: not opt -passes=verify -S %s 2>&1 | FileCheck %s
; Check that verifier doesn't crash on relocate with none token

target triple = "x86_64-unknown-linux-gnu"

define i32 @check_verify_none_token() gc "statepoint-example" {

entry:
    ret i32 0

unreach:
    ; CHECK: gc relocate is incorrectly tied to the statepoint
    ; CHECK: (undef, undef)
    %token_call = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token none, i32 0, i32 0)
    ret i32 1
}

declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token, i32, i32)
