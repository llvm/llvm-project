; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s
;
; Ensures that volatile stores/loads count as MemoryDefs

; CHECK-LABEL: define i32 @foo
define i32 @foo() {
  %1 = alloca i32, align 4
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store volatile i32 4
  store volatile i32 4, i32* %1, align 4
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: store volatile i32 8
  store volatile i32 8, i32* %1, align 4
; CHECK: 3 = MemoryDef(2)
; CHECK-NEXT: %2 = load volatile i32
  %2 = load volatile i32, i32* %1, align 4
; CHECK: 4 = MemoryDef(3)
; CHECK-NEXT: %3 = load volatile i32
  %3 = load volatile i32, i32* %1, align 4
  %4 = add i32 %3, %2
  ret i32 %4
}

; Ensuring we allow hoisting nonvolatile loads around volatile loads.
; CHECK-LABEL: define void @volatile_only
define void @volatile_only(i32* %arg1, i32* %arg2) {
  ; Trivially NoAlias/MustAlias
  %a = alloca i32
  %b = alloca i32

; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: load volatile i32, i32* %a
  load volatile i32, i32* %a
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: load i32, i32* %b
  load i32, i32* %b
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: load i32, i32* %a
  load i32, i32* %a

  ; MayAlias
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: load volatile i32, i32* %arg1
  load volatile i32, i32* %arg1
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: load i32, i32* %arg2
  load i32, i32* %arg2

  ret void
}

; Ensuring that volatile atomic operations work properly.
; CHECK-LABEL: define void @volatile_atomics
define void @volatile_atomics(i32* %arg1, i32* %arg2) {
  %a = alloca i32
  %b = alloca i32

 ; Trivially NoAlias/MustAlias

; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: load atomic volatile i32, i32* %a acquire, align 4
  load atomic volatile i32, i32* %a acquire, align 4
; CHECK: MemoryUse(1)
; CHECK-NEXT: load i32, i32* %b
  load i32, i32* %b

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: load atomic volatile i32, i32* %a monotonic, align 4
  load atomic volatile i32, i32* %a monotonic, align 4
; CHECK: MemoryUse(1)
; CHECK-NEXT: load i32, i32* %b
  load i32, i32* %b
; CHECK: MemoryUse(1)
; CHECK-NEXT: load atomic i32, i32* %b unordered, align 4
  load atomic i32, i32* %b unordered, align 4
; CHECK: MemoryUse(1)
; CHECK-NEXT: load atomic i32, i32* %a unordered, align 4
  load atomic i32, i32* %a unordered, align 4
; CHECK: MemoryUse(1)
; CHECK-NEXT: load i32, i32* %a
  load i32, i32* %a

  ; MayAlias
; CHECK: 3 = MemoryDef(2)
; CHECK-NEXT: load atomic volatile i32, i32* %arg1 monotonic, align 4
  load atomic volatile i32, i32* %arg1 monotonic, align 4
; CHECK: MemoryUse(1)
; CHECK-NEXT: load i32, i32* %arg2
  load i32, i32* %arg2

  ret void
}
