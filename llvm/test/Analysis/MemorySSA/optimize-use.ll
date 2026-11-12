; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NOLIMIT
; RUN: opt -memssa-check-limit=0 -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,LIMIT

; Function Attrs: ssp uwtable
define i32 @main() {
entry:
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT:   %call = call noalias ptr @_Znwm(i64 4)
  %call = call noalias ptr @_Znwm(i64 4)
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT:   %call1 = call noalias ptr @_Znwm(i64 4)
  %call1 = call noalias ptr @_Znwm(i64 4)
; CHECK: 3 = MemoryDef(2)
; CHECK-NEXT:   store i32 5, ptr %call, align 4
  store i32 5, ptr %call, align 4
; CHECK: 4 = MemoryDef(3)
; CHECK-NEXT:   store i32 7, ptr %call1, align 4
  store i32 7, ptr %call1, align 4
; NOLIMIT: MemoryUse(3)
; NOLIMIT-NEXT:   %0 = load i32, ptr %call, align 4
; LIMIT: MemoryUse(4)
; LIMIT-NEXT:   %0 = load i32, ptr %call, align 4
  %0 = load i32, ptr %call, align 4
; NOLIMIT: MemoryUse(4)
; NOLIMIT-NEXT:   %1 = load i32, ptr %call1, align 4
; LIMIT: MemoryUse(4)
; LIMIT-NEXT:   %1 = load i32, ptr %call1, align 4
  %1 = load i32, ptr %call1, align 4
; NOLIMIT: MemoryUse(3)
; NOLIMIT-NEXT:   %2 = load i32, ptr %call, align 4
; LIMIT: MemoryUse(4)
; LIMIT-NEXT:   %2 = load i32, ptr %call, align 4
  %2 = load i32, ptr %call, align 4
; NOLIMIT: MemoryUse(4)
; NOLIMIT-NEXT:   %3 = load i32, ptr %call1, align 4
; LIMIT: MemoryUse(4)
; LIMIT-NEXT:   %3 = load i32, ptr %call1, align 4
  %3 = load i32, ptr %call1, align 4
  %add = add nsw i32 %1, %3
  ret i32 %add
}


declare noalias ptr @_Znwm(i64)
