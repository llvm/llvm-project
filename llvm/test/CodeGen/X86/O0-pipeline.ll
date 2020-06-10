; When EXPENSIVE_CHECKS are enabled, the machine verifier appears between each
; pass. Ignore it with 'grep -v'.
; RUN: llc -mtriple=x86_64-- -O0 -debug-pass=Structure < %s -o /dev/null 2>&1 \
; RUN:   | grep -v 'Verify generated machine code' | FileCheck %s

; REQUIRES: asserts

; CHECK-LABEL: Pass Arguments:
; CHECK-NEXT: Target Library Information
; CHECK-NEXT: Target Pass Configuration
; CHECK-NEXT: Machine Module Information
; CHECK-NEXT: Target Transform Information
; CHECK-NEXT: Type-Based Alias Analysis
; CHECK-NEXT: Scoped NoAlias Alias Analysis
; CHECK-NEXT: Assumption Cache Tracker
; CHECK-NEXT: Create Garbage Collector Module Metadata
; CHECK-NEXT: Profile summary info
; CHECK-NEXT: Machine Branch Probability Analysis
; CHECK-NEXT:   ModulePass Manager
; CHECK-NEXT:     Pre-ISel Intrinsic Lowering
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Expand Atomic instructions
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:       Module Verifier
; CHECK-NEXT:       Lower Garbage Collection Instructions
; CHECK-NEXT:       Shadow Stack GC Lowering
; CHECK-NEXT:       Lower constant intrinsics
; CHECK-NEXT:       Remove unreachable blocks from the CFG
; CHECK-NEXT:       Instrument function entry/exit with calls to e.g. mcount() (post inlining)
; CHECK-NEXT:       Scalarize Masked Memory Intrinsics
; CHECK-NEXT:       Expand reduction intrinsics
; CHECK-NEXT:       Expand indirectbr instructions
; CHECK-NEXT:     Rewrite Symbols
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Exception handling preparation
; CHECK-NEXT:       Safe Stack instrumentation pass
; CHECK-NEXT:       Insert stack protectors
; CHECK-NEXT:       Module Verifier
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       X86 DAG->DAG Instruction Selection
; CHECK-NEXT:       X86 PIC Global Base Reg Initialization
; CHECK-NEXT:       Finalize ISel and expand pseudo-instructions
; CHECK-NEXT:       Local Stack Slot Allocation
; CHECK-NEXT:       X86 speculative load hardening
; CHECK-NEXT:       MachineDominator Tree Construction
; CHECK-NEXT:       X86 EFLAGS copy lowering
; CHECK-NEXT:       X86 WinAlloca Expander
; CHECK-NEXT:       Eliminate PHI nodes for register allocation
; CHECK-NEXT:       Two-Address instruction pass
; CHECK-NEXT:       Fast Register Allocator
; CHECK-NEXT:       Bundle Machine CFG Edges
; CHECK-NEXT:       X86 FP Stackifier
; CHECK-NEXT:       X86 Load Value Injection (LVI) Load Hardening (Unoptimized)
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Machine Optimization Remark Emitter
; CHECK-NEXT:       Prologue/Epilogue Insertion & Frame Finalization
; CHECK-NEXT:       Post-RA pseudo instruction expansion pass
; CHECK-NEXT:       X86 pseudo instruction expansion pass
; CHECK-NEXT:       Analyze Machine Code For Garbage Collection
; CHECK-NEXT:       Insert fentry calls
; CHECK-NEXT:       Insert XRay ops
; CHECK-NEXT:       Implement the 'patchable-function' attribute
; CHECK-NEXT:       X86 Indirect Branch Tracking
; CHECK-NEXT:       X86 vzeroupper inserter
; CHECK-NEXT:       X86 Discriminate Memory Operands
; CHECK-NEXT:       X86 Insert Cache Prefetches
; CHECK-NEXT:       Contiguously Lay Out Funclets
; CHECK-NEXT:       StackMap Liveness Analysis
; CHECK-NEXT:       Live DEBUG_VALUE analysis
; CHECK-NEXT:       X86 Indirect Thunks
; CHECK-NEXT:       Check CFA info and insert CFI instructions if needed
; CHECK-NEXT:       X86 Load Value Injection (LVI) Ret-Hardening
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Machine Optimization Remark Emitter
; CHECK-NEXT:       X86 Assembly Printer
; CHECK-NEXT:       Free MachineFunction

define void @f() {
  ret void
}

