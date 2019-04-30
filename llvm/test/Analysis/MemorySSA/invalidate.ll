; RUN: opt -aa-pipeline=basic-aa -passes='require<memoryssa>,invalidate<aa>,early-cse-memssa' \
; RUN:     -debug-pass-manager -disable-output %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-AA-INVALIDATE
; RUN: opt -aa-pipeline=basic-aa -passes='require<memoryssa>,invalidate<domtree>,early-cse-memssa' \
; RUN:     -debug-pass-manager -disable-output %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-DT-INVALIDATE

; CHECK-AA-INVALIDATE: Running analysis: MemorySSAAnalysis
; CHECK-AA-INVALIDATE: Running analysis: DominatorTreeAnalysis
; CHECK-AA-INVALIDATE: Running analysis: AAManager
; CHECK-AA-INVALIDATE: Running analysis: BasicAA
; CHECK-AA-INVALIDATE: Running pass: InvalidateAnalysisPass<llvm::AAManager>
; CHECK-AA-INVALIDATE: Invalidating analysis: AAManager
; CHECK-AA-INVALIDATE: Invalidating analysis: MemorySSAAnalysis
; CHECK-AA-INVALIDATE: Running pass: EarlyCSEPass
; CHECK-AA-INVALIDATE: Running analysis: MemorySSAAnalysis
; CHECK-AA-INVALIDATE: Running analysis: AAManager

; CHECK-DT-INVALIDATE: Running analysis: MemorySSAAnalysis
; CHECK-DT-INVALIDATE: Running analysis: DominatorTreeAnalysis
; CHECK-DT-INVALIDATE: Running analysis: AAManager
; CHECK-DT-INVALIDATE: Running analysis: BasicAA
; CHECK-DT-INVALIDATE: InvalidateAnalysisPass<llvm::DominatorTreeAnalysis>
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
  %call = call noalias i8* @_Znwm(i64 4)
  %0 = bitcast i8* %call to i32*
  %call1 = call noalias i8* @_Znwm(i64 4)
  %1 = bitcast i8* %call1 to i32*
  store i32 5, i32* %0, align 4
  store i32 7, i32* %1, align 4
  %2 = load i32, i32* %0, align 4
  %3 = load i32, i32* %1, align 4
  %4 = load i32, i32* %0, align 4
  %5 = load i32, i32* %1, align 4
  %add = add nsw i32 %3, %5
  ret i32 %add
}

declare noalias i8* @_Znwm(i64)

