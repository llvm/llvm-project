; RUN: opt -prune-eh -S < %s | FileCheck %s

declare void @noreturn()

define void @testfn() {
    ; A musttail call must be followed by (optional bitcast then) ret,
    ; so make sure we don't insert an unreachable
    ; CHECK: musttail call void @noreturn
    ; CHECK-NOT: unreachable
    ; CHECK-NEXT: ret void
    musttail call void @noreturn() #0
    ret void
}

attributes #0 = { noreturn }
