; RUN: llc < %s -mcpu=corei7 | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

;CHECK-LABEL: ltstore:
;CHECK: movq
;CHECK: movq
;CHECK: ret
define void @ltstore(ptr %pA, ptr %pB) {
entry:
  %in = load <4 x i32>, ptr %pA
  %j = shufflevector <4 x i32> %in, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  store <2 x i32> %j, ptr %pB
  ret void
}

