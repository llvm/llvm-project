; RUN: opt < %s -passes='dse,print<memoryssa>' -disable-output 2>&1 | FileCheck %s --check-prefix=MSSA
; RUN: opt < %s -passes='dse,early-cse<memssa>' -S | FileCheck %s

; DSE reports MemorySSA as preserved. When it deletes a MemoryDef that was an
; incoming value of a MemoryPhi, the remaining operands can become identical
; and leave the phi trivial. A stale phi is still valid, but it is returned as
; the clobber for later queries and blocks the next MemorySSA consumer.

define void @dead_store_on_one_arm(i1 %c, ptr %p, ptr %q, i1 %v) {
; The phi in %join is trivial once the store in %right is gone, so it should
; not survive into the preserved MemorySSA.
; MSSA-LABEL: MemorySSA for function: dead_store_on_one_arm
; MSSA-NOT:   MemoryPhi
;
; CHECK-LABEL: define void @dead_store_on_one_arm(
; CHECK:       join:
; CHECK-NEXT:    ret void
;
entry:
  store i1 %v, ptr %q
  %x = load i32, ptr %p
  br i1 %c, label %left, label %right

left:
  br label %join

right:
  store i1 %v, ptr %q
  br label %join

join:
  store i32 %x, ptr %p
  ret void
}
