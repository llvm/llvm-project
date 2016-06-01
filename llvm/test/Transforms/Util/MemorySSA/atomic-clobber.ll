; RUN: opt -basicaa -memoryssa -analyze < %s 2>&1 | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s
;
; Ensures that atomic loads count as MemoryDefs

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
