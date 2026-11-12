; RUN: opt -passes='print<memoryssa-walker>' -disable-output < %s 2>&1 | FileCheck %s

; CHECK: define void @test
; CHECK: 1 = MemoryDef(liveOnEntry)->liveOnEntry - clobbered by liveOnEntry
; CHECK: store i8 42, ptr %a1
; CHECK: 2 = MemoryDef(1)->liveOnEntry - clobbered by liveOnEntry
; CHECK: store i8 42, ptr %a2
; CHECK: MemoryUse(1) - clobbered by 1 = MemoryDef(liveOnEntry)->liveOnEntry
; CHECK: %l1 = load i8, ptr %a1
; CHECK: MemoryUse(2) - clobbered by 2 = MemoryDef(1)->liveOnEntry
; CHECK: %l2 = load i8, ptr %a2
; CHECK: 3 = MemoryDef(2)->liveOnEntry - clobbered by liveOnEntry
; CHECK: store i8 42, ptr %p
; CHECK: 4 = MemoryDef(3)->3 - clobbered by 3 = MemoryDef(2)->liveOnEntry
; CHECK: store i8 42, ptr %p
; CHECK: MemoryUse(4) - clobbered by 4 = MemoryDef(3)->3
; CHECK: %p1 = load i8, ptr %p
; CHECK: MemoryUse(4) - clobbered by 4 = MemoryDef(3)->3
; CHECK: %p2 = load i8, ptr %p

define void @test(ptr %p) {
  %a1 = alloca i8
  %a2 = alloca i8
  store i8 42, ptr %a1
  store i8 42, ptr %a2
  %l1 =  load i8, ptr %a1
  %l2 =  load i8, ptr %a2

  store i8 42, ptr %p
  store i8 42, ptr %p
  %p1 =  load i8, ptr %p
  %p2 =  load i8, ptr %p

  ret void
}
