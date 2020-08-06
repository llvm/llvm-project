; RUN: opt -basic-aa -print-memoryssa -verify-memoryssa -enable-new-pm=0 -analyze < %s 2>&1 | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s
;
; Ensures that atomic loads count as MemoryDefs

; CHECK-LABEL: define i32 @foo
define i32 @foo(i32* %a, i32* %b) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 4
  store i32 4, i32* %a, align 4
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: %1 = load atomic i32
  %1 = load atomic i32, i32* %b acquire, align 4
; CHECK: MemoryUse(2)
; CHECK-NEXT: %2 = load i32
  %2 = load i32, i32* %a, align 4
  %3 = add i32 %1, %2
  ret i32 %3
}

; CHECK-LABEL: define void @bar
define void @bar(i32* %a) {
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: load atomic i32, i32* %a unordered, align 4
  load atomic i32, i32* %a unordered, align 4
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: load atomic i32, i32* %a monotonic, align 4
  load atomic i32, i32* %a monotonic, align 4
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: load atomic i32, i32* %a acquire, align 4
  load atomic i32, i32* %a acquire, align 4
; CHECK: 3 = MemoryDef(2)
; CHECK-NEXT: load atomic i32, i32* %a seq_cst, align 4
  load atomic i32, i32* %a seq_cst, align 4
  ret void
}

; CHECK-LABEL: define void @baz
define void @baz(i32* %a) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: %1 = load atomic i32
  %1 = load atomic i32, i32* %a acquire, align 4
; CHECK: MemoryUse(1)
; CHECK-NEXT: %2 = load atomic i32, i32* %a unordered, align 4
  %2 = load atomic i32, i32* %a unordered, align 4
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: %3 = load atomic i32, i32* %a monotonic, align 4
  %3 = load atomic i32, i32* %a monotonic, align 4
  ret void
}

; CHECK-LABEL: define void @fences
define void @fences(i32* %a) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: fence acquire
  fence acquire
; CHECK: MemoryUse(1)
; CHECK-NEXT: %1 = load i32, i32* %a
  %1 = load i32, i32* %a

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: fence release
  fence release
; CHECK: MemoryUse(2)
; CHECK-NEXT: %2 = load i32, i32* %a
  %2 = load i32, i32* %a

; CHECK: 3 = MemoryDef(2)
; CHECK-NEXT: fence acq_rel
  fence acq_rel
; CHECK: MemoryUse(3)
; CHECK-NEXT: %3 = load i32, i32* %a
  %3 = load i32, i32* %a

; CHECK: 4 = MemoryDef(3)
; CHECK-NEXT: fence seq_cst
  fence seq_cst
; CHECK: MemoryUse(4)
; CHECK-NEXT: %4 = load i32, i32* %a
  %4 = load i32, i32* %a
  ret void
}

; CHECK-LABEL: define void @seq_cst_clobber
define void @seq_cst_clobber(i32* noalias %a, i32* noalias %b) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: %1 = load atomic i32, i32* %a monotonic, align 4
  load atomic i32, i32* %a monotonic, align 4

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: %2 = load atomic i32, i32* %a seq_cst, align 4
  load atomic i32, i32* %a seq_cst, align 4

; CHECK: 3 = MemoryDef(2)
; CHECK-NEXT: load atomic i32, i32* %a monotonic, align 4
  load atomic i32, i32* %a monotonic, align 4

  ret void
}

; Ensure that AA hands us MRI_Mod on unreorderable atomic ops.
;
; This test is a bit implementation-specific. In particular, it depends on that
; we pass cmpxchg-load queries to AA, without trying to reason about them on
; our own.
;
; If AA gets more aggressive, we can find another way.
;
; CHECK-LABEL: define void @check_aa_is_sane
define void @check_aa_is_sane(i32* noalias %a, i32* noalias %b) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: cmpxchg i32* %a, i32 0, i32 1 acquire acquire
  cmpxchg i32* %a, i32 0, i32 1 acquire acquire
; CHECK: MemoryUse(1)
; CHECK-NEXT: load i32, i32* %b, align 4
  load i32, i32* %b, align 4

  ret void
}
