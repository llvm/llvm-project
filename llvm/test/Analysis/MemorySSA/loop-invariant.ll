; RUN: opt -passes='print<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s

; TODO: The load's MemoryUse can be defined by liveOnEntry. Since
; %p2 is a loop invariant and the MemoryLoc of load instr and store inst in
; loop block are NoAlias
;
; CHECK: MemoryUse(2)
; CHECK: %val = load i32, ptr %p2
define void @gep(ptr %ptr) {
entry:
  %p1 = getelementptr i32, ptr %ptr, i32 1
  br label %tmp

tmp:
  %p2 = getelementptr i32, ptr %p1, i32 1
  br label %loop

loop:
  %x = phi i32 [ 0, %tmp ], [ %x.inc, %loop ]
  %val = load i32, ptr %p2
  %p3 = getelementptr i32, ptr %p2, i32 1
  store volatile i32 0, ptr %p3
  %x.inc = add i32 %x, %val
  br label %loop
}

; CHECK: MemoryUse(2)
; CHECK-NEXT: %val = load i32, ptr %p2
define void @load_entry_block(ptr %ptr, ptr %addr) {
entry:
  %p1 = load ptr, ptr %ptr
  br label %tmp

tmp:
  %p2 = getelementptr i32, ptr %p1, i32 1
  br label %loop

loop:
  %x = phi i32 [ 0, %tmp ], [ %x.inc, %loop ]
  %val = load i32, ptr %p2
  %p3 = getelementptr i32, ptr %p2, i32 1
  store volatile i32 0, ptr %p3
  %x.inc = add i32 %x, %val
  br label %loop
}
