; RUN: opt -basicaa -print-memoryssa -analyze -verify-memoryssa < %s 2>&1 | FileCheck %s

; Function Attrs: ssp uwtable
define i32 @main() {
entry:
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT:   %call = call noalias i8* @_Znwm(i64 4)
  %call = call noalias i8* @_Znwm(i64 4)
  %0 = bitcast i8* %call to i32*
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT:   %call1 = call noalias i8* @_Znwm(i64 4)
  %call1 = call noalias i8* @_Znwm(i64 4)
  %1 = bitcast i8* %call1 to i32*
; CHECK: 3 = MemoryDef(2)
; CHECK-NEXT:   store i32 5, i32* %0, align 4
  store i32 5, i32* %0, align 4
; CHECK: 4 = MemoryDef(3)
; CHECK-NEXT:   store i32 7, i32* %1, align 4
  store i32 7, i32* %1, align 4
; CHECK: MemoryUse(3)
; CHECK-NEXT:   %2 = load i32, i32* %0, align 4
  %2 = load i32, i32* %0, align 4
; CHECK: MemoryUse(4)
; CHECK-NEXT:   %3 = load i32, i32* %1, align 4
  %3 = load i32, i32* %1, align 4
; CHECK: MemoryUse(3)
; CHECK-NEXT:   %4 = load i32, i32* %0, align 4
  %4 = load i32, i32* %0, align 4
; CHECK: MemoryUse(4)
; CHECK-NEXT:   %5 = load i32, i32* %1, align 4
  %5 = load i32, i32* %1, align 4
  %add = add nsw i32 %3, %5
  ret i32 %add
}

declare noalias i8* @_Znwm(i64)
