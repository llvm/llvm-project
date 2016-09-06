; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

define void @test() {
entry:
  br i1 undef, label %split1, label %split2

split1:
  store i16 undef, i16* undef, align 2
 br label %merge
split2:
 br label %merge
forwardunreachable:
  br label %merge
merge:
; The forwardunreachable block still needs an entry in the phi node,
; because it is reverse reachable, so the CFG still has it as a
; predecessor of the block
; CHECK:  3 = MemoryPhi({split1,1},{split2,liveOnEntry},{forwardunreachable,liveOnEntry})
  store i16 undef, i16* undef, align 2
  ret void
}

