; RUN: opt -passes='require<no-op-module>,inline' < %s -S -debug-pass-manager 2>&1 | FileCheck %s
; RUN: opt -passes='require<no-op-module>,cgscc(devirt<1>(inline))' < %s -S -debug-pass-manager 2>&1 | FileCheck %s

; CHECK: Invalidating analysis: NoOpModuleAnalysis

; CHECK: ModuleID
; CHECK-NOT: define

define linkonce_odr i32 @f(i1 %0) {
  br i1 %0, label %2, label %4

2:                                                ; preds = %1
  %3 = call i32 @f(i1 false)
  br label %4

4:                                                ; preds = %2, %1
  ret i32 0
}

