; RUN: opt -disable-output -passes="loop-mssa(loop-rotate),print<memoryssa>" -verify-memoryssa < %s 2>&1 | FileCheck %s

; CHECK: entry:
; CHECK-NEXT: 3 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store ptr null, ptr %p, align 8
; CHECK-NEXT: MemoryUse(3)
; CHECK-NEXT: %val11 = load ptr, ptr %p, align 8

; CHECK: loop.latch:
; CHECK-NEXT: 5 = MemoryPhi({loop.latch,1},{loop.latch.lr.ph,3})
; CHECK-NEXT: MemoryUse(5)
; CHECK-NEXT: %val2 = load ptr, ptr %p, align 8
; CHECK-NEXT: 1 = MemoryDef(5)
; CHECK-NEXT: store ptr null, ptr %p, align 8
; CHECK-NEXT: MemoryUse(1)
; CHECK-NEXT: %val1 = load ptr, ptr %p, align 8

; CHECK: exit:
; CHECK-NEXT: 4 = MemoryPhi({entry,3},{loop.exit_crit_edge,1})

define void @test(ptr %p) {
entry:
  br label %loop

loop:
  store ptr null, ptr %p
  %val1 = load ptr, ptr %p
  %cmp = icmp eq ptr %val1, null
  br i1 %cmp, label %exit, label %loop.latch

loop.latch:
  %val2 = load ptr, ptr %p
  br label %loop

exit:
  ret void
}
