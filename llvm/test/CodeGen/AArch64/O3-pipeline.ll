; RUN: llc --debugify-and-strip-all-safe=0 -mtriple=arm64-- -O3 -debug-pass=Structure < %s -o /dev/null 2>&1 | \
; RUN:     grep -v "Verify generated machine code" | FileCheck %s

; REQUIRES: asserts

; CHECK-LABEL: Pass Arguments:
; CHECK-NEXT: Target Library Information
; CHECK-NEXT: Target Pass Configuration
; CHECK-NEXT: Machine Module Information
; CHECK-NEXT: Target Transform Information
; CHECK-NEXT: Assumption Cache Tracker
; CHECK-NEXT: Profile summary info
; CHECK-NEXT: Type-Based Alias Analysis
; CHECK-NEXT: Scoped NoAlias Alias Analysis
; CHECK-NEXT: Create Garbage Collector Module Metadata
; CHECK-NEXT: Machine Branch Probability Analysis
; CHECK-NEXT: Default Regalloc Eviction Advisor
; CHECK-NEXT: Default Regalloc Priority Advisor
; CHECK-NEXT:   ModulePass Manager
; CHECK-NEXT:     Pre-ISel Intrinsic Lowering
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Expand large div/rem
; CHECK-NEXT:       Expand Atomic instructions
; CHECK-NEXT:     SVE intrinsics optimizations
; CHECK-NEXT:       FunctionPass Manager
; CHECK-NEXT:         Dominator Tree Construction
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Simplify the CFG
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       Canonicalize natural loops
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       Optimization Remark Emitter
; CHECK-NEXT:       Scalar Evolution Analysis
; CHECK-NEXT:       Loop Data Prefetch
; CHECK-NEXT:       Falkor HW Prefetch Fix
; CHECK-NEXT:       Split GEPs to a variadic base and a constant offset for better CSE
; CHECK-NEXT:       Early CSE
; CHECK-NEXT:       Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       Memory SSA
; CHECK-NEXT:       Canonicalize natural loops
; CHECK-NEXT:       LCSSA Verifier
; CHECK-NEXT:       Loop-Closed SSA Form Pass
; CHECK-NEXT:       Scalar Evolution Analysis
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       Loop Pass Manager
; CHECK-NEXT:         Loop Invariant Code Motion
; CHECK-NEXT:       Module Verifier
; CHECK-NEXT:       Loop Pass Manager
; CHECK-NEXT:         Canonicalize Freeze Instructions in Loops
; CHECK-NEXT:         Induction Variable Users
; CHECK-NEXT:         Loop Strength Reduction
; CHECK-NEXT:       Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       Merge contiguous icmps into a memcmp
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       Expand memcmp() to load/stores
; CHECK-NEXT:       Lower Garbage Collection Instructions
; CHECK-NEXT:       Shadow Stack GC Lowering
; CHECK-NEXT:       Lower constant intrinsics
; CHECK-NEXT:       Remove unreachable blocks from the CFG
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       Post-Dominator Tree Construction
; CHECK-NEXT:       Branch Probability Analysis
; CHECK-NEXT:       Block Frequency Analysis
; CHECK-NEXT:       Constant Hoisting
; CHECK-NEXT:       Replace intrinsics with calls to vector library
; CHECK-NEXT:       Partially inline calls to library functions
; CHECK-NEXT:       Expand vector predication intrinsics
; CHECK-NEXT:       Scalarize Masked Memory Intrinsics
; CHECK-NEXT:       Expand reduction intrinsics
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       TLS Variable Hoist
; CHECK-NEXT:     Stack Safety Analysis
; CHECK-NEXT:       FunctionPass Manager
; CHECK-NEXT:         Dominator Tree Construction
; CHECK-NEXT:         Natural Loop Information
; CHECK-NEXT:         Scalar Evolution Analysis
; CHECK-NEXT:         Stack Safety Local Analysis
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       AArch64 Stack Tagging
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       Memory SSA
; CHECK-NEXT:       Interleaved Load Combine Pass
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Interleaved Access Pass
; CHECK-NEXT:       SME ABI Pass
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       Type Promotion
; CHECK-NEXT:       CodeGen Prepare
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Exception handling preparation
; CHECK-NEXT:     AArch64 Promote Constant
; CHECK-NEXT:       FunctionPass Manager
; CHECK-NEXT:         Dominator Tree Construction
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Merge internal globals
; CHECK-NEXT:       Safe Stack instrumentation pass
; CHECK-NEXT:       Insert stack protectors
; CHECK-NEXT:       Module Verifier
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       Post-Dominator Tree Construction
; CHECK-NEXT:       Branch Probability Analysis
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       AArch64 Instruction Selection
; CHECK-NEXT:       MachineDominator Tree Construction
; CHECK-NEXT:       AArch64 Local Dynamic TLS Access Clean-up
; CHECK-NEXT:       Finalize ISel and expand pseudo-instructions
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Early Tail Duplication
; CHECK-NEXT:       Optimize machine instruction PHIs
; CHECK-NEXT:       Slot index numbering
; CHECK-NEXT:       Merge disjoint stack slots
; CHECK-NEXT:       Local Stack Slot Allocation
; CHECK-NEXT:       Remove dead machine instructions
; CHECK-NEXT:       MachineDominator Tree Construction
; CHECK-NEXT:       AArch64 Condition Optimizer
; CHECK-NEXT:       Machine Natural Loop Construction
; CHECK-NEXT:       Machine Trace Metrics
; CHECK-NEXT:       AArch64 Conditional Compares
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Machine InstCombiner
; CHECK-NEXT:       AArch64 Conditional Branch Tuning
; CHECK-NEXT:       Machine Trace Metrics
; CHECK-NEXT:       Early If-Conversion
; CHECK-NEXT:       AArch64 Store Pair Suppression
; CHECK-NEXT:       AArch64 SIMD instructions optimization pass
; CHECK-NEXT:       AArch64 Stack Tagging PreRA
; CHECK-NEXT:       MachineDominator Tree Construction
; CHECK-NEXT:       Machine Natural Loop Construction
; CHECK-NEXT:       Machine Block Frequency Analysis
; CHECK-NEXT:       Early Machine Loop Invariant Code Motion
; CHECK-NEXT:       MachineDominator Tree Construction
; CHECK-NEXT:       Machine Block Frequency Analysis
; CHECK-NEXT:       Machine Common Subexpression Elimination
; CHECK-NEXT:       MachinePostDominator Tree Construction
; CHECK-NEXT:       Machine Cycle Info Analysis
; CHECK-NEXT:       Machine code sinking
; CHECK-NEXT:       Peephole Optimizations
; CHECK-NEXT:       Remove dead machine instructions
; CHECK-NEXT:       AArch64 MI Peephole Optimization pass
; CHECK-NEXT:       AArch64 Dead register definitions
; CHECK-NEXT:       Detect Dead Lanes
; CHECK-NEXT:       Process Implicit Definitions
; CHECK-NEXT:       Remove unreachable machine basic blocks
; CHECK-NEXT:       Live Variable Analysis
; CHECK-NEXT:       Eliminate PHI nodes for register allocation
; CHECK-NEXT:       Two-Address instruction pass
; CHECK-NEXT:       MachineDominator Tree Construction
; CHECK-NEXT:       Slot index numbering
; CHECK-NEXT:       Live Interval Analysis
; CHECK-NEXT:       Simple Register Coalescing
; CHECK-NEXT:       Rename Disconnected Subregister Components
; CHECK-NEXT:       Machine Instruction Scheduler
; CHECK-NEXT:       Machine Block Frequency Analysis
; CHECK-NEXT:       Debug Variable Analysis
; CHECK-NEXT:       Live Stack Slot Analysis
; CHECK-NEXT:       Virtual Register Map
; CHECK-NEXT:       Live Register Matrix
; CHECK-NEXT:       Bundle Machine CFG Edges
; CHECK-NEXT:       Spill Code Placement Analysis
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Machine Optimization Remark Emitter
; CHECK-NEXT:       Greedy Register Allocator
; CHECK-NEXT:       Virtual Register Rewriter
; CHECK-NEXT:       Register Allocation Pass Scoring
; CHECK-NEXT:       Stack Slot Coloring
; CHECK-NEXT:       Machine Copy Propagation Pass
; CHECK-NEXT:       Machine Loop Invariant Code Motion
; CHECK-NEXT:       AArch64 Redundant Copy Elimination
; CHECK-NEXT:       A57 FP Anti-dependency breaker
; CHECK-NEXT:       Remove Redundant DEBUG_VALUE analysis
; CHECK-NEXT:       Fixup Statepoint Caller Saved
; CHECK-NEXT:       PostRA Machine Sink
; CHECK-NEXT:       MachineDominator Tree Construction
; CHECK-NEXT:       Machine Natural Loop Construction
; CHECK-NEXT:       Machine Block Frequency Analysis
; CHECK-NEXT:       MachinePostDominator Tree Construction
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Machine Optimization Remark Emitter
; CHECK-NEXT:       Shrink Wrapping analysis
; CHECK-NEXT:       Prologue/Epilogue Insertion & Frame Finalization
; CHECK-NEXT:       Control Flow Optimizer
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Tail Duplication
; CHECK-NEXT:       Machine Copy Propagation Pass
; CHECK-NEXT:       Post-RA pseudo instruction expansion pass
; CHECK-NEXT:       AArch64 pseudo instruction expansion pass
; CHECK-NEXT:       AArch64 load / store optimization pass
; CHECK-NEXT:       Insert KCFI indirect call checks
; CHECK-NEXT:       AArch64 speculation hardening pass
; CHECK-NEXT:       AArch64 Indirect Thunks
; CHECK-NEXT:       AArch64 sls hardening pass
; CHECK-NEXT:       MachineDominator Tree Construction
; CHECK-NEXT:       Machine Natural Loop Construction
; CHECK-NEXT:       Falkor HW Prefetch Fix Late Phase
; CHECK-NEXT:       PostRA Machine Instruction Scheduler
; CHECK-NEXT:       Analyze Machine Code For Garbage Collection
; CHECK-NEXT:       Machine Block Frequency Analysis
; CHECK-NEXT:       MachinePostDominator Tree Construction
; CHECK-NEXT:       Branch Probability Basic Block Placement
; CHECK-NEXT:       Insert fentry calls
; CHECK-NEXT:       Insert XRay ops
; CHECK-NEXT:       Implement the 'patchable-function' attribute
; CHECK-NEXT:       AArch64 load / store optimization pass
; CHECK-NEXT:       Machine Copy Propagation Pass
; CHECK-NEXT:       Workaround A53 erratum 835769 pass
; CHECK-NEXT:       AArch64 Branch Targets
; CHECK-NEXT:       Branch relaxation pass
; CHECK-NEXT:       AArch64 Compress Jump Tables
; CHECK-NEXT:       Contiguously Lay Out Funclets
; CHECK-NEXT:       StackMap Liveness Analysis
; CHECK-NEXT:       Live DEBUG_VALUE analysis
; CHECK-NEXT:     Machine Outliner
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Insert CFI remember/restore state instructions
; CHECK-NEXT:       Unpack machine instruction bundles
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Machine Optimization Remark Emitter
; CHECK-NEXT:       AArch64 Assembly Printer
; CHECK-NEXT:       Free MachineFunction
; CHECK-NEXT: Pass Arguments:  -domtree
; CHECK-NEXT:   FunctionPass Manager
; CHECK-NEXT:     Dominator Tree Construction
; CHECK-NEXT: Pass Arguments:  -assumption-cache-tracker -targetlibinfo -domtree -loops -scalar-evolution -stack-safety-local
; CHECK-NEXT: Assumption Cache Tracker
; CHECK-NEXT: Target Library Information
; CHECK-NEXT:   FunctionPass Manager
; CHECK-NEXT:     Dominator Tree Construction
; CHECK-NEXT:     Natural Loop Information
; CHECK-NEXT:     Scalar Evolution Analysis
; CHECK-NEXT:     Stack Safety Local Analysis
; CHECK-NEXT: Pass Arguments:  -domtree
; CHECK-NEXT:   FunctionPass Manager
; CHECK-NEXT:     Dominator Tree Construction

define void @f() {
  ret void
}
