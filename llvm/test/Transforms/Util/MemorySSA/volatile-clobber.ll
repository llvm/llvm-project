; RUN: opt -basicaa -print-memoryssa -verify-memoryssa -analyze < %s 2>&1 | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s
;
; Ensures that volatile stores/loads count as MemoryDefs

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
