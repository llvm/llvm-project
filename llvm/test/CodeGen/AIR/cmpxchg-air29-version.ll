; The compare-and-exchange with memory flags only exists from AIR 2.9 (MSL
; 4.1). A module that uses it gets air.version 2.9 above the target's, and a
; warning: a system without MSL 4.1 rejects it ("expecting 2.8").
; RUN: llc -mtriple=air64_v28-apple-macosx26.0.0 %s -o - 2>&1 | FileCheck %s

; CHECK: warning: air.atomic.global.cmpxchg.weak.i32 with memory flags needs AIR 2.9 (MSL 4.1)
; CHECK: !air.version = !{![[V:[0-9]+]]}
; CHECK: ![[V]] = !{i32 2, i32 9, i32 0}

declare i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1), ptr, i32, i32, i32, i32, i32, i1)

define void @kernel(ptr addrspace(1) %p) {
entry:
  %exp = alloca i32
  store i32 0, ptr %exp
  %old = call i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1) %p, ptr %exp, i32 1, i32 0, i32 0, i32 2, i32 1, i1 true)
  ret void
}
