; RUN: opt -aa-pipeline=basic-aa -passes='require<memoryssa>,invalidate<aa>,early-cse<memssa>' \
; RUN:     -debug-pass-manager -disable-output %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-AA-INVALIDATE
; RUN: opt -aa-pipeline=basic-aa -passes='require<memoryssa>,invalidate<domtree>,early-cse<memssa>' \
; RUN:     -debug-pass-manager -disable-output %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-DT-INVALIDATE

; CHECK-AA-INVALIDATE: Running analysis: MemorySSAAnalysis
; CHECK-AA-INVALIDATE: Running analysis: DominatorTreeAnalysis
; CHECK-AA-INVALIDATE: Running analysis: AAManager
; CHECK-AA-INVALIDATE: Running analysis: BasicAA
; CHECK-AA-INVALIDATE: Running pass: InvalidateAnalysisPass<{{.*}}AAManager
; CHECK-AA-INVALIDATE: Invalidating analysis: AAManager
; CHECK-AA-INVALIDATE: Invalidating analysis: MemorySSAAnalysis
; CHECK-AA-INVALIDATE: Running pass: EarlyCSEPass
; CHECK-AA-INVALIDATE: Running analysis: MemorySSAAnalysis
; CHECK-AA-INVALIDATE: Running analysis: AAManager

; CHECK-DT-INVALIDATE: Running analysis: MemorySSAAnalysis
; CHECK-DT-INVALIDATE: Running analysis: DominatorTreeAnalysis
; CHECK-DT-INVALIDATE: Running analysis: AAManager
; CHECK-DT-INVALIDATE: Running analysis: BasicAA
; CHECK-DT-INVALIDATE: Running pass: InvalidateAnalysisPass<{{.*}}DominatorTreeAnalysis
; CHECK-DT-INVALIDATE: Invalidating analysis: DominatorTreeAnalysis
; CHECK-DT-INVALIDATE: Invalidating analysis: BasicAA
; CHECK-DT-INVALIDATE: Invalidating analysis: AAManager
; CHECK-DT-INVALIDATE: Invalidating analysis: MemorySSAAnalysis
; CHECK-DT-INVALIDATE: Running pass: EarlyCSEPass
; CHECK-DT-INVALIDATE: Running analysis: DominatorTreeAnalysis
; CHECK-DT-INVALIDATE: Running analysis: MemorySSAAnalysis
; CHECK-DT-INVALIDATE: Running analysis: AAManager
; CHECK-DT-INVALIDATE: Running analysis: BasicAA


; Function Attrs: ssp uwtable
define i32 @main() {
entry:
  %call = call noalias ptr @_Znwm(i64 4)
  %call1 = call noalias ptr @_Znwm(i64 4)
  store i32 5, ptr %call, align 4
  store i32 7, ptr %call1, align 4
  %0 = load i32, ptr %call, align 4
  %1 = load i32, ptr %call1, align 4
  %2 = load i32, ptr %call, align 4
  %3 = load i32, ptr %call1, align 4
  %add = add nsw i32 %1, %3
  ret i32 %add
}

declare noalias ptr @_Znwm(i64)

