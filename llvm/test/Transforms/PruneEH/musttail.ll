; RUN: opt -prune-eh -enable-new-pm=0 -S < %s | FileCheck %s
; RUN: opt < %s -passes='function-attrs,function(simplifycfg)' -S | FileCheck %s

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
