; RUN: llc < %s -mtriple=arm64-apple-ios7.0 -pass-remarks-analysis=asm-printer \
; RUN:       -pass-remarks-with-hotness=1 -asm-verbose=0 \
; RUN:       -debug-only=lazy-machine-block-freq,block-freq \
; RUN:       -debug-pass=Executions 2>&1 | FileCheck %s -check-prefix=HOTNESS

; RUN: llc < %s -mtriple=arm64-apple-ios7.0 -pass-remarks-analysis=asm-printer \
; RUN:       -pass-remarks-with-hotness=0 -asm-verbose=0 \
; RUN:       -debug-only=lazy-machine-block-freq,block-freq \
; RUN:       -debug-pass=Executions 2>&1 | FileCheck %s -check-prefix=NO_HOTNESS

; REQUIRES: asserts


; Verify that we don't new populate MachineBFI for passes that already use
; MBFI, e.g. GreedyRegAlloc.  (This hard-codes the previous pass to the
; GreedyRegAlloc, please adjust accordingly.)

; HOTNESS:      Executing Pass 'Spill Code Placement Analysis'
; HOTNESS-NEXT: Executing Pass 'Lazy Machine Block Frequency Analysis'
; HOTNESS-NEXT: Executing Pass 'Machine Optimization Remark Emitter'
; HOTNESS-NEXT: MachineBlockFrequencyInfo is available
; HOTNESS-NEXT: Executing Pass 'Greedy Register Allocator'


; Verify that we only populate MachineBFI on behalf of ORE when hotness is
; requested.  (This hard-codes the previous pass to the Assembly Printer,
; please adjust accordingly.)

; HOTNESS:      Executing Pass 'Implement the 'patchable-function' attribute'
; HOTNESS-NEXT:  Freeing Pass 'Implement the 'patchable-function' attribute'
; HOTNESS-NEXT: Executing Pass 'Lazy Machine Block Frequency Analysis'
; HOTNESS-NEXT: Executing Pass 'Machine Optimization Remark Emitter'
; HOTNESS-NEXT: Building MachineBlockFrequencyInfo on the fly
; HOTNESS-NEXT: Building LoopInfo on the fly
; HOTNESS-NEXT: Building DominatorTree on the fly
; HOTNESS-NOT: Executing Pass
; HOTNESS: block-frequency: empty_func
; HOTNESS-NOT: Executing Pass
; HOTNESS: Executing Pass 'MachineDominator Tree Construction'
; HOTNESS-NEXT: Executing Pass 'Machine Natural Loop Construction'
; HOTNESS-NEXT: Executing Pass 'AArch64 Assembly Printer'

; HOTNESS: arm64-summary-remarks.ll:5:0: 1 instructions in function (hotness: 33)


; NO_HOTNESS:      Executing Pass 'Implement the 'patchable-function' attribute'
; NO_HOTNESS-NEXT:  Freeing Pass 'Implement the 'patchable-function' attribute'
; NO_HOTNESS-NEXT: Executing Pass 'Lazy Machine Block Frequency Analysis'
; NO_HOTNESS-NEXT: Executing Pass 'Machine Optimization Remark Emitter'
; NO_HOTNESS-NEXT: Executing Pass 'MachineDominator Tree Construction'
; NO_HOTNESS-NEXT: Executing Pass 'Machine Natural Loop Construction'
; NO_HOTNESS-NEXT: Executing Pass 'AArch64 Assembly Printer'

; NO_HOTNESS: arm64-summary-remarks.ll:5:0: 1 instructions in function{{$}}

define void @empty_func() nounwind ssp !dbg !3 !prof !4 {
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1)
!1 = !DIFile(filename: "arm64-summary-remarks.ll", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "empty_func", scope: !1, file: !1, line: 5, scopeLine: 5, unit: !0)
!4 = !{!"function_entry_count", i64 33}
