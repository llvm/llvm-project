; RUN: opt -basicaa -print-memoryssa -verify-memoryssa -analyze < %s 2>&1 | FileCheck %s
;
; Currently, MemorySSA doesn't support invariant groups. So, we should ignore
; invariant.group.barrier intrinsics entirely. We'll need to pay attention to
; them when/if we decide to support invariant groups.

@g = external global i32

define i32 @foo(i32* %a) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, i32* %a, align 4, !invariant.group !0

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: store i32 1
  store i32 1, i32* @g, align 4

  %1 = bitcast i32* %a to i8*
  %a8 = call i8* @llvm.invariant.group.barrier(i8* %1)
  %a32 = bitcast i8* %a8 to i32*

; This have to be MemoryUse(1), because we can't skip the barrier based on
; invariant.group.
; CHECK: MemoryUse(2)
; CHECK-NEXT: %2 = load i32
  %2 = load i32, i32* %a32, align 4, !invariant.group !0
  ret i32 %2
}

define i32 @skipBarrier(i32* %a) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, i32* %a, align 4, !invariant.group !0

  %1 = bitcast i32* %a to i8*
  %a8 = call i8* @llvm.invariant.group.barrier(i8* %1)
  %a32 = bitcast i8* %a8 to i32*

; We can skip the barrier only if the "skip" is not based on !invariant.group.
; CHECK: MemoryUse(1)
; CHECK-NEXT: %2 = load i32
  %2 = load i32, i32* %a32, align 4, !invariant.group !0
  ret i32 %2
}

define i32 @skipBarrier2(i32* %a) {

; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %v = load i32
  %v = load i32, i32* %a, align 4, !invariant.group !0

  %1 = bitcast i32* %a to i8*
  %a8 = call i8* @llvm.invariant.group.barrier(i8* %1)
  %a32 = bitcast i8* %a8 to i32*

; We can skip the barrier only if the "skip" is not based on !invariant.group.
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %v2 = load i32
  %v2 = load i32, i32* %a32, align 4, !invariant.group !0
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 1
  store i32 1, i32* @g, align 4

; FIXME: based on invariant.group it should be MemoryUse(liveOnEntry)
; CHECK: MemoryUse(1)
; CHECK-NEXT: %v3 = load i32
  %v3 = load i32, i32* %a32, align 4, !invariant.group !0
  %add = add nsw i32 %v2, %v3
  %add2 = add nsw i32 %add, %v
  ret i32 %add2
}

define i32 @handleInvariantGroups(i32* %a) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, i32* %a, align 4, !invariant.group !0

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: store i32 1
  store i32 1, i32* @g, align 4
  %1 = bitcast i32* %a to i8*
  %a8 = call i8* @llvm.invariant.group.barrier(i8* %1)
  %a32 = bitcast i8* %a8 to i32*

; CHECK: MemoryUse(2)
; CHECK-NEXT: %2 = load i32
  %2 = load i32, i32* %a32, align 4, !invariant.group !0

; CHECK: 3 = MemoryDef(2)
; CHECK-NEXT: store i32 2
  store i32 2, i32* @g, align 4

; FIXME: This can be changed to MemoryUse(2)
; CHECK: MemoryUse(3)
; CHECK-NEXT: %3 = load i32
  %3 = load i32, i32* %a32, align 4, !invariant.group !0
  %add = add nsw i32 %2, %3
  ret i32 %add
}

declare i8* @llvm.invariant.group.barrier(i8*)

!0 = !{!"group1"}
