; RUN: opt -basicaa -print-memoryssa -verify-memoryssa -analyze < %s 2>&1 | FileCheck %s
;
; Ensuring that external functions without attributes are MemoryDefs

@g = external global i32
declare void @modifyG()

define i32 @foo() {
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %1 = load i32
  %1 = load i32, i32* @g

; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 4
  store i32 4, i32* @g, align 4

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @modifyG()
  call void @modifyG()

; CHECK: MemoryUse(2)
; CHECK-NEXT: %2 = load i32
  %2 = load i32, i32* @g
  %3 = add i32 %2, %1
  ret i32 %3
}
