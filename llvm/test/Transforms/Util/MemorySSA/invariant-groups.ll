; RUN: opt -basicaa -print-memoryssa -verify-memoryssa -analyze < %s 2>&1 | FileCheck %s
;
; Currently, MemorySSA doesn't support invariant groups. So, we should ignore
; invariant.group.barrier intrinsics entirely. We'll need to pay attention to
; them when/if we decide to support invariant groups.

@g = external global i32

define i32 @foo(i32* %a) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, i32* %a, align 4, !llvm.invariant.group !0

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: store i32 1
  store i32 1, i32* @g, align 4

  %1 = bitcast i32* %a to i8*
  %a8 = call i8* @llvm.invariant.group.barrier(i8* %1)
  %a32 = bitcast i8* %a8 to i32*

; CHECK: MemoryUse(2)
; CHECK-NEXT: %2 = load i32
  %2 = load i32, i32* %a32, align 4, !llvm.invariant.group !0
  ret i32 %2
}

declare i8* @llvm.invariant.group.barrier(i8*)

!0 = !{!"group1"}
