; RUN: opt -S -passes=verify < %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

define void @foo() gc "statepoint_example" personality ptr @P {
; CHECK-NOT: gc.result operand #1 must be from a statepoint   
entry:
    br label %label_1
label_1:
    ; CHECK: ret void
    ret void

label_2:
    ; CHECK: token poison
    %call = call noundef i32 @llvm.experimental.gc.result.i32(token poison)
    unreachable
}   

declare i32 @llvm.experimental.gc.result.i32(token)

declare ptr @P()
