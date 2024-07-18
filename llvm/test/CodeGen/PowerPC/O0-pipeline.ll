; RUN: llc --debugify-and-strip-all-safe=0 -mtriple=powerpc64-- -O0 \
; RUN:   -debug-pass=Structure < %s -o /dev/null 2>&1 | \
; RUN:   grep -v "Verify generated machine code" | FileCheck %s

; REQUIRES: asserts

; CHECK-LABEL: Pass Arguments:
; CHECK-NEXT: Target Library Information
; CHECK-NEXT: Target Pass Configuration
; CHECK-NEXT: Machine Module Information
; CHECK-NEXT: Target Transform Information
; CHECK-NEXT: Create Garbage Collector Module Metadata
; CHECK-NEXT: Assumption Cache Tracker
; CHECK-NEXT: Profile summary info
; CHECK-NEXT: Machine Branch Probability Analysis
; CHECK-NEXT:   ModulePass Manager
; CHECK-NEXT:     Pre-ISel Intrinsic Lowering
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Expand large div/rem
; CHECK-NEXT:       Expand large fp convert
; CHECK-NEXT:       Expand Atomic instructions
; CHECK-NEXT:     PPC Lower MASS Entries
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Module Verifier
; CHECK-NEXT:       Lower Garbage Collection Instructions
; CHECK-NEXT:       Shadow Stack GC Lowering
; CHECK-NEXT:       Lower constant intrinsics
; CHECK-NEXT:       Remove unreachable blocks from the CFG
; CHECK-NEXT:       Expand vector predication intrinsics
; CHECK-NEXT:       Instrument function entry/exit with calls to e.g. mcount() (post inlining)
; CHECK-NEXT:       Scalarize Masked Memory Intrinsics
; CHECK-NEXT:       Expand reduction intrinsics
; CHECK-NEXT:       Exception handling preparation
; CHECK-NEXT:       Prepare callbr
; CHECK-NEXT:       Safe Stack instrumentation pass
; CHECK-NEXT:       Insert stack protectors
; CHECK-NEXT:       Module Verifier
; CHECK-NEXT:       Assignment Tracking Analysis
; CHECK-NEXT:       PowerPC DAG->DAG Pattern Instruction Selection
; CHECK-NEXT:       PowerPC VSX Copy Legalization
; CHECK-NEXT:       Finalize ISel and expand pseudo-instructions
; CHECK-NEXT:       Local Stack Slot Allocation
; CHECK-NEXT:       Remove unreachable machine basic blocks
; CHECK-NEXT:       Live Variable Analysis
; CHECK-NEXT:       MachineDominator Tree Construction
; CHECK-NEXT:       Slot index numbering
; CHECK-NEXT:       Live Interval Analysis
; CHECK-NEXT:       PowerPC TLS Dynamic Call Fixup
; CHECK-NEXT:       PowerPC TOC Register Dependencies
; CHECK-NEXT:       Eliminate PHI nodes for register allocation
; CHECK-NEXT:       Two-Address instruction pass
; CHECK-NEXT:       Fast Register Allocator
; CHECK-NEXT:       Remove Redundant DEBUG_VALUE analysis
; CHECK-NEXT:       Fixup Statepoint Caller Saved
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Machine Optimization Remark Emitter
; CHECK-NEXT:       Prologue/Epilogue Insertion & Frame Finalization
; CHECK-NEXT:       Post-RA pseudo instruction expansion pass
; CHECK-NEXT:       Analyze Machine Code For Garbage Collection
; CHECK-NEXT:       Insert fentry calls
; CHECK-NEXT:       Insert XRay ops
; CHECK-NEXT:       Implement the 'patchable-function' attribute
; CHECK-NEXT:       PowerPC Pre-Emit Peephole
; CHECK-NEXT:       PowerPC Expand ISEL Generation
; CHECK-NEXT:       Contiguously Lay Out Funclets
; CHECK-NEXT:       StackMap Liveness Analysis
; CHECK-NEXT:       Live DEBUG_VALUE analysis
; CHECK-NEXT:       Machine Sanitizer Binary Metadata
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Machine Optimization Remark Emitter
; CHECK-NEXT:       Stack Frame Layout Analysis
; CHECK-NEXT:       PowerPC Expand Atomic
; CHECK-NEXT:       PowerPC Branch Selector
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Machine Optimization Remark Emitter
; CHECK-NEXT:       Linux PPC Assembly Printer
; CHECK-NEXT:       Free MachineFunction

define void @f() {
  ret void
}
