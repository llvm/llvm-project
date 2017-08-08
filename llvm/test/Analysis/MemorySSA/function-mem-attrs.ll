; RUN: opt -basicaa -print-memoryssa -verify-memoryssa -analyze < %s 2>&1 | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s
;
; Test that various function attributes give us sane results.

@g = external global i32

declare void @readonlyFunction() readonly
declare void @noattrsFunction()

define void @readonlyAttr() {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, i32* @g, align 4

  %1 = alloca i32, align 4
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: store i32 0
  store i32 0, i32* %1, align 4

; CHECK: MemoryUse(1)
; CHECK-NEXT: call void @readonlyFunction()
  call void @readonlyFunction()

; CHECK: MemoryUse(1)
; CHECK-NEXT: call void @noattrsFunction() #
; Assume that #N is readonly
  call void @noattrsFunction() readonly

  ; Sanity check that noattrsFunction is otherwise a MemoryDef
; CHECK: 3 = MemoryDef(2)
; CHECK-NEXT: call void @noattrsFunction()
  call void @noattrsFunction()
  ret void
}

declare void @argMemOnly(i32*) argmemonly

define void @inaccessableOnlyAttr() {
  %1 = alloca i32, align 4
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, i32* %1, align 4

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: store i32 0
  store i32 0, i32* @g, align 4

; CHECK: MemoryUse(1)
; CHECK-NEXT: call void @argMemOnly(i32* %1) #
; Assume that #N is readonly
  call void @argMemOnly(i32* %1) readonly

; CHECK: 3 = MemoryDef(2)
; CHECK-NEXT: call void @argMemOnly(i32* %1)
  call void @argMemOnly(i32* %1)

  ret void
}
