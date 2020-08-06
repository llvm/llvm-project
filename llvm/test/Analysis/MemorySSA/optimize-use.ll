; RUN: opt -basic-aa -print-memoryssa -verify-memoryssa -enable-new-pm=0 -analyze < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NOLIMIT
; RUN: opt -memssa-check-limit=0 -basic-aa -print-memoryssa -verify-memoryssa -enable-new-pm=0 -analyze < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,LIMIT
; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NOLIMIT
; RUN: opt -memssa-check-limit=0 -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,LIMIT

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
; NOLIMIT: MemoryUse(3) MustAlias
; NOLIMIT-NEXT:   %2 = load i32, i32* %0, align 4
; LIMIT: MemoryUse(4) MayAlias
; LIMIT-NEXT:   %2 = load i32, i32* %0, align 4
  %2 = load i32, i32* %0, align 4
; NOLIMIT: MemoryUse(4) MustAlias
; NOLIMIT-NEXT:   %3 = load i32, i32* %1, align 4
; LIMIT: MemoryUse(4) MayAlias
; LIMIT-NEXT:   %3 = load i32, i32* %1, align 4
  %3 = load i32, i32* %1, align 4
; NOLIMIT: MemoryUse(3) MustAlias
; NOLIMIT-NEXT:   %4 = load i32, i32* %0, align 4
; LIMIT: MemoryUse(4) MayAlias
; LIMIT-NEXT:   %4 = load i32, i32* %0, align 4
  %4 = load i32, i32* %0, align 4
; NOLIMIT: MemoryUse(4) MustAlias
; NOLIMIT-NEXT:   %5 = load i32, i32* %1, align 4
; LIMIT: MemoryUse(4) MayAlias
; LIMIT-NEXT:   %5 = load i32, i32* %1, align 4
  %5 = load i32, i32* %1, align 4
  %add = add nsw i32 %3, %5
  ret i32 %add
}


declare noalias i8* @_Znwm(i64)
