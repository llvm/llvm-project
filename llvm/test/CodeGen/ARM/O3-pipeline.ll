; RUN: llc -mtriple=arm -O3 -debug-pass=Structure < %s -o /dev/null 2>&1 | grep -v "Verify generated machine code" | FileCheck %s

; REQUIRES: asserts

; CHECK:       ModulePass Manager
; CHECK-NEXT:    Pre-ISel Intrinsic Lowering
; CHECK-NEXT:    FunctionPass Manager
; CHECK-NEXT:      Expand Atomic instructions
; CHECK-NEXT:      Simplify the CFG
; CHECK-NEXT:      MVE gather/scatter lowering
; CHECK-NEXT:      Dominator Tree Construction
; CHECK-NEXT:      Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:      Module Verifier
; CHECK-NEXT:      Natural Loop Information
; CHECK-NEXT:      Canonicalize natural loops
; CHECK-NEXT:      Scalar Evolution Analysis
; CHECK-NEXT:      Loop Pass Manager
; CHECK-NEXT:        Induction Variable Users
; CHECK-NEXT:        Loop Strength Reduction
; CHECK-NEXT:      Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:      Function Alias Analysis Results
; CHECK-NEXT:      Merge contiguous icmps into a memcmp
; CHECK-NEXT:      Natural Loop Information
; CHECK-NEXT:      Lazy Branch Probability Analysis
; CHECK-NEXT:      Lazy Block Frequency Analysis
; CHECK-NEXT:      Expand memcmp() to load/stores
; CHECK-NEXT:      Lower Garbage Collection Instructions
; CHECK-NEXT:      Shadow Stack GC Lowering
; CHECK-NEXT:      Lower constant intrinsics
; CHECK-NEXT:      Remove unreachable blocks from the CFG
; CHECK-NEXT:      Dominator Tree Construction
; CHECK-NEXT:      Natural Loop Information
; CHECK-NEXT:      Branch Probability Analysis
; CHECK-NEXT:      Block Frequency Analysis
; CHECK-NEXT:      Constant Hoisting
; CHECK-NEXT:      Partially inline calls to library functions
; CHECK-NEXT:      Instrument function entry/exit with calls to e.g. mcount() (post inlining)
; CHECK-NEXT:      Scalarize Masked Memory Intrinsics
; CHECK-NEXT:      Expand reduction intrinsics
; CHECK-NEXT:      Dominator Tree Construction
; CHECK-NEXT:      Natural Loop Information
; CHECK-NEXT:      Scalar Evolution Analysis
; CHECK-NEXT:      Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:      Function Alias Analysis Results
; CHECK-NEXT:      Transform functions to use DSP intrinsics
; CHECK-NEXT:      Interleaved Access Pass
; CHECK-NEXT:      Type Promotion
; CHECK-NEXT:      Dominator Tree Construction
; CHECK-NEXT:      Natural Loop Information
; CHECK-NEXT:      CodeGen Prepare
; CHECK-NEXT:    Rewrite Symbols
; CHECK-NEXT:    FunctionPass Manager
; CHECK-NEXT:      Dominator Tree Construction
; CHECK-NEXT:      Exception handling preparation
; CHECK-NEXT:      Merge internal globals
; CHECK-NEXT:      Dominator Tree Construction
; CHECK-NEXT:      Natural Loop Information
; CHECK-NEXT:      Scalar Evolution Analysis
; CHECK-NEXT:      Lazy Branch Probability Analysis
; CHECK-NEXT:      Lazy Block Frequency Analysis
; CHECK-NEXT:      Optimization Remark Emitter
; CHECK-NEXT:      Hardware Loop Insertion
; CHECK-NEXT:      Scalar Evolution Analysis
; CHECK-NEXT:      Loop Pass Manager
; CHECK-NEXT:        Transform predicated vector loops to use MVE tail predication
; CHECK-NEXT:      Safe Stack instrumentation pass
; CHECK-NEXT:      Insert stack protectors
; CHECK-NEXT:      Module Verifier
; CHECK-NEXT:      Dominator Tree Construction
; CHECK-NEXT:      Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:      Function Alias Analysis Results
; CHECK-NEXT:      Natural Loop Information
; CHECK-NEXT:      Branch Probability Analysis
; CHECK-NEXT:      Lazy Branch Probability Analysis
; CHECK-NEXT:      Lazy Block Frequency Analysis
; CHECK-NEXT:      ARM Instruction Selection
; CHECK-NEXT:      Finalize ISel and expand pseudo-instructions
; CHECK-NEXT:      Lazy Machine Block Frequency Analysis
; CHECK-NEXT:      Early Tail Duplication
; CHECK-NEXT:      Optimize machine instruction PHIs
; CHECK-NEXT:      Slot index numbering
; CHECK-NEXT:      Merge disjoint stack slots
; CHECK-NEXT:      Local Stack Slot Allocation
; CHECK-NEXT:      Remove dead machine instructions
; CHECK-NEXT:      MachineDominator Tree Construction
; CHECK-NEXT:      Machine Natural Loop Construction
; CHECK-NEXT:      Early Machine Loop Invariant Code Motion
; CHECK-NEXT:      MachineDominator Tree Construction
; CHECK-NEXT:      Machine Block Frequency Analysis
; CHECK-NEXT:      Machine Common Subexpression Elimination
; CHECK-NEXT:      MachinePostDominator Tree Construction
; CHECK-NEXT:      Machine code sinking
; CHECK-NEXT:      Peephole Optimizations
; CHECK-NEXT:      Remove dead machine instructions
; CHECK-NEXT:      ARM MLA / MLS expansion pass
; CHECK-NEXT:      ARM pre- register allocation load / store optimization pass
; CHECK-NEXT:      ARM A15 S->D optimizer
; CHECK-NEXT:      Detect Dead Lanes
; CHECK-NEXT:      Process Implicit Definitions
; CHECK-NEXT:      Remove unreachable machine basic blocks
; CHECK-NEXT:      Live Variable Analysis
; CHECK-NEXT:      MachineDominator Tree Construction
; CHECK-NEXT:      Machine Natural Loop Construction
; CHECK-NEXT:      Eliminate PHI nodes for register allocation
; CHECK-NEXT:      Two-Address instruction pass
; CHECK-NEXT:      Slot index numbering
; CHECK-NEXT:      Live Interval Analysis
; CHECK-NEXT:      Simple Register Coalescing
; CHECK-NEXT:      Rename Disconnected Subregister Components
; CHECK-NEXT:      Machine Instruction Scheduler
; CHECK-NEXT:      Machine Block Frequency Analysis
; CHECK-NEXT:      Debug Variable Analysis
; CHECK-NEXT:      Live Stack Slot Analysis
; CHECK-NEXT:      Virtual Register Map
; CHECK-NEXT:      Live Register Matrix
; CHECK-NEXT:      Bundle Machine CFG Edges
; CHECK-NEXT:      Spill Code Placement Analysis
; CHECK-NEXT:      Lazy Machine Block Frequency Analysis
; CHECK-NEXT:      Machine Optimization Remark Emitter
; CHECK-NEXT:      Greedy Register Allocator
; CHECK-NEXT:      Virtual Register Rewriter
; CHECK-NEXT:      Stack Slot Coloring
; CHECK-NEXT:      Machine Copy Propagation Pass
; CHECK-NEXT:      Machine Loop Invariant Code Motion
; CHECK-NEXT:      PostRA Machine Sink
; CHECK-NEXT:      Machine Block Frequency Analysis
; CHECK-NEXT:      MachineDominator Tree Construction
; CHECK-NEXT:      MachinePostDominator Tree Construction
; CHECK-NEXT:      Lazy Machine Block Frequency Analysis
; CHECK-NEXT:      Machine Optimization Remark Emitter
; CHECK-NEXT:      Shrink Wrapping analysis
; CHECK-NEXT:      Prologue/Epilogue Insertion & Frame Finalization
; CHECK-NEXT:      Control Flow Optimizer
; CHECK-NEXT:      Lazy Machine Block Frequency Analysis
; CHECK-NEXT:      Tail Duplication
; CHECK-NEXT:      Machine Copy Propagation Pass
; CHECK-NEXT:      Post-RA pseudo instruction expansion pass
; CHECK-NEXT:      ARM load / store optimization pass
; CHECK-NEXT:      ReachingDefAnalysis
; CHECK-NEXT:      ARM Execution Domain Fix
; CHECK-NEXT:      BreakFalseDeps
; CHECK-NEXT:      ARM pseudo instruction expansion pass
; CHECK-NEXT:      Thumb2 instruction size reduce pass
; CHECK-NEXT:      MachineDominator Tree Construction
; CHECK-NEXT:      Machine Natural Loop Construction
; CHECK-NEXT:      Machine Block Frequency Analysis
; CHECK-NEXT:      If Converter
; CHECK-NEXT:      MVE VPT block insertion pass
; CHECK-NEXT:      Thumb IT blocks insertion pass
; CHECK-NEXT:      MachineDominator Tree Construction
; CHECK-NEXT:      Machine Natural Loop Construction
; CHECK-NEXT:      PostRA Machine Instruction Scheduler
; CHECK-NEXT:      Post RA top-down list latency scheduler
; CHECK-NEXT:      Analyze Machine Code For Garbage Collection
; CHECK-NEXT:      Machine Block Frequency Analysis
; CHECK-NEXT:      MachinePostDominator Tree Construction
; CHECK-NEXT:      Branch Probability Basic Block Placement
; CHECK-NEXT:      Insert fentry calls
; CHECK-NEXT:      Insert XRay ops
; CHECK-NEXT:      Implement the 'patchable-function' attribute
; CHECK-NEXT:      Thumb2 instruction size reduce pass
; CHECK-NEXT:      Unpack machine instruction bundles
; CHECK-NEXT:      optimise barriers pass
; CHECK-NEXT:      Contiguously Lay Out Funclets
; CHECK-NEXT:      StackMap Liveness Analysis
; CHECK-NEXT:      Live DEBUG_VALUE analysis
; CHECK-NEXT:      MachineDominator Tree Construction
; CHECK-NEXT:      ARM constant island placement and branch shortening pass
; CHECK-NEXT:      MachineDominator Tree Construction
; CHECK-NEXT:      Machine Natural Loop Construction
; CHECK-NEXT:      ReachingDefAnalysis
; CHECK-NEXT:      ARM Low Overhead Loops pass
; CHECK-NEXT:      Lazy Machine Block Frequency Analysis
; CHECK-NEXT:      Machine Optimization Remark Emitter
; CHECK-NEXT:      ARM Assembly Printer
; CHECK-NEXT:      Free MachineFunction
