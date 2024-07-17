;; When EXPENSIVE_CHECKS are enabled, the machine verifier appears between each
;; pass. Ignore it with 'grep -v'.
; RUN: llc --mtriple=loongarch32 -mattr=+d -O1 --debug-pass=Structure %s -o /dev/null 2>&1 | \
; RUN:   grep -v "Verify generated machine code" | FileCheck %s --check-prefix=LAXX
; RUN: llc --mtriple=loongarch32 -mattr=+d -O2 --debug-pass=Structure %s -o /dev/null 2>&1 | \
; RUN:   grep -v "Verify generated machine code" | FileCheck %s --check-prefix=LAXX
; RUN: llc --mtriple=loongarch32 -mattr=+d -O3 --debug-pass=Structure %s -o /dev/null 2>&1 | \
; RUN:   grep -v "Verify generated machine code" | FileCheck %s --check-prefix=LAXX
; RUN: llc --mtriple=loongarch64 -mattr=+d -O1 --debug-pass=Structure %s -o /dev/null 2>&1 | \
; RUN:   grep -v "Verify generated machine code" | FileCheck %s --check-prefixes=LAXX,LA64
; RUN: llc --mtriple=loongarch64 -mattr=+d -O2 --debug-pass=Structure %s -o /dev/null 2>&1 | \
; RUN:   grep -v "Verify generated machine code" | FileCheck %s --check-prefixes=LAXX,LA64
; RUN: llc --mtriple=loongarch64 -mattr=+d -O3 --debug-pass=Structure %s -o /dev/null 2>&1 | \
; RUN:   grep -v "Verify generated machine code" | FileCheck %s --check-prefixes=LAXX,LA64

; REQUIRES: asserts

; LAXX-LABEL: Pass Arguments:
; LAXX-NEXT: Target Library Information
; LAXX-NEXT: Target Pass Configuration
; LAXX-NEXT: Machine Module Information
; LAXX-NEXT: Target Transform Information
; LAXX-NEXT: Type-Based Alias Analysis
; LAXX-NEXT: Scoped NoAlias Alias Analysis
; LAXX-NEXT: Assumption Cache Tracker
; LAXX-NEXT: Profile summary info
; LAXX-NEXT: Create Garbage Collector Module Metadata
; LAXX-NEXT: Machine Branch Probability Analysis
; LAXX-NEXT: Default Regalloc Eviction Advisor
; LAXX-NEXT: Default Regalloc Priority Advisor
; LAXX-NEXT:   ModulePass Manager
; LAXX-NEXT:     Pre-ISel Intrinsic Lowering
; LAXX-NEXT:     FunctionPass Manager
; LAXX-NEXT:       Expand large div/rem
; LAXX-NEXT:       Expand large fp convert
; LAXX-NEXT:       Expand Atomic instructions
; LAXX-NEXT:       Module Verifier
; LAXX-NEXT:       Dominator Tree Construction
; LAXX-NEXT:       Basic Alias Analysis (stateless AA impl)
; LAXX-NEXT:       Natural Loop Information
; LAXX-NEXT:       Canonicalize natural loops
; LAXX-NEXT:       Scalar Evolution Analysis
; LAXX-NEXT:       Loop Pass Manager
; LAXX-NEXT:         Canonicalize Freeze Instructions in Loops
; LAXX-NEXT:         Induction Variable Users
; LAXX-NEXT:         Loop Strength Reduction
; LAXX-NEXT:       Basic Alias Analysis (stateless AA impl)
; LAXX-NEXT:       Function Alias Analysis Results
; LAXX-NEXT:       Merge contiguous icmps into a memcmp
; LAXX-NEXT:       Natural Loop Information
; LAXX-NEXT:       Lazy Branch Probability Analysis
; LAXX-NEXT:       Lazy Block Frequency Analysis
; LAXX-NEXT:       Expand memcmp() to load/stores
; LAXX-NEXT:       Lower Garbage Collection Instructions
; LAXX-NEXT:       Shadow Stack GC Lowering
; LAXX-NEXT:       Lower constant intrinsics
; LAXX-NEXT:       Remove unreachable blocks from the CFG
; LAXX-NEXT:       Natural Loop Information
; LAXX-NEXT:       Post-Dominator Tree Construction
; LAXX-NEXT:       Branch Probability Analysis
; LAXX-NEXT:       Block Frequency Analysis
; LAXX-NEXT:       Constant Hoisting
; LAXX-NEXT:       Replace intrinsics with calls to vector library
; LAXX-NEXT:       Partially inline calls to library functions
; LAXX-NEXT:       Expand vector predication intrinsics
; LAXX-NEXT:       Instrument function entry/exit with calls to e.g. mcount() (post inlining)
; LAXX-NEXT:       Scalarize Masked Memory Intrinsics
; LAXX-NEXT:       Expand reduction intrinsics
; LAXX-NEXT:       Natural Loop Information
; LAXX-NEXT:       TLS Variable Hoist
; LAXX-NEXT:       CodeGen Prepare
; LAXX-NEXT:       Dominator Tree Construction
; LAXX-NEXT:       Exception handling preparation
; LAXX-NEXT:       Prepare callbr
; LAXX-NEXT:       Safe Stack instrumentation pass
; LAXX-NEXT:       Insert stack protectors
; LAXX-NEXT:       Module Verifier
; LAXX-NEXT:       Basic Alias Analysis (stateless AA impl)
; LAXX-NEXT:       Function Alias Analysis Results
; LAXX-NEXT:       Natural Loop Information
; LAXX-NEXT:       Post-Dominator Tree Construction
; LAXX-NEXT:       Branch Probability Analysis
; LAXX-NEXT:       Assignment Tracking Analysis
; LAXX-NEXT:       Lazy Branch Probability Analysis
; LAXX-NEXT:       Lazy Block Frequency Analysis
; LAXX-NEXT:       LoongArch DAG->DAG Pattern Instruction Selection
; LAXX-NEXT:       Finalize ISel and expand pseudo-instructions
; LAXX-NEXT:       Lazy Machine Block Frequency Analysis
; LAXX-NEXT:       Early Tail Duplication
; LAXX-NEXT:       Optimize machine instruction PHIs
; LAXX-NEXT:       Slot index numbering
; LAXX-NEXT:       Merge disjoint stack slots
; LAXX-NEXT:       Local Stack Slot Allocation
; LAXX-NEXT:       Remove dead machine instructions
; LAXX-NEXT:       MachineDominator Tree Construction
; LAXX-NEXT:       Machine Natural Loop Construction
; LAXX-NEXT:       Machine Block Frequency Analysis
; LAXX-NEXT:       Early Machine Loop Invariant Code Motion
; LAXX-NEXT:       MachineDominator Tree Construction
; LAXX-NEXT:       Machine Block Frequency Analysis
; LAXX-NEXT:       Machine Common Subexpression Elimination
; LAXX-NEXT:       MachinePostDominator Tree Construction
; LAXX-NEXT:       Machine Cycle Info Analysis
; LAXX-NEXT:       Machine code sinking
; LAXX-NEXT:       Peephole Optimizations
; LAXX-NEXT:       Remove dead machine instructions
; LA64-NEXT:       LoongArch Optimize W Instructions
; LAXX-NEXT:       LoongArch Pre-RA pseudo instruction expansion pass
; LAXX-NEXT:       Detect Dead Lanes
; LAXX-NEXT:       Init Undef Pass
; LAXX-NEXT:       Process Implicit Definitions
; LAXX-NEXT:       Remove unreachable machine basic blocks
; LAXX-NEXT:       Live Variable Analysis
; LAXX-NEXT:       Eliminate PHI nodes for register allocation
; LAXX-NEXT:       Two-Address instruction pass
; LAXX-NEXT:       MachineDominator Tree Construction
; LAXX-NEXT:       Slot index numbering
; LAXX-NEXT:       Live Interval Analysis
; LAXX-NEXT:       Register Coalescer
; LAXX-NEXT:       Rename Disconnected Subregister Components
; LAXX-NEXT:       Machine Instruction Scheduler
; LAXX-NEXT:       LoongArch Dead register definitions
; LAXX-NEXT:       Machine Block Frequency Analysis
; LAXX-NEXT:       Debug Variable Analysis
; LAXX-NEXT:       Live Stack Slot Analysis
; LAXX-NEXT:       Virtual Register Map
; LAXX-NEXT:       Live Register Matrix
; LAXX-NEXT:       Bundle Machine CFG Edges
; LAXX-NEXT:       Spill Code Placement Analysis
; LAXX-NEXT:       Lazy Machine Block Frequency Analysis
; LAXX-NEXT:       Machine Optimization Remark Emitter
; LAXX-NEXT:       Greedy Register Allocator
; LAXX-NEXT:       Virtual Register Rewriter
; LAXX-NEXT:       Register Allocation Pass Scoring
; LAXX-NEXT:       Stack Slot Coloring
; LAXX-NEXT:       Machine Copy Propagation Pass
; LAXX-NEXT:       Machine Loop Invariant Code Motion
; LAXX-NEXT:       Remove Redundant DEBUG_VALUE analysis
; LAXX-NEXT:       Fixup Statepoint Caller Saved
; LAXX-NEXT:       PostRA Machine Sink
; LAXX-NEXT:       Machine Block Frequency Analysis
; LAXX-NEXT:       MachineDominator Tree Construction
; LAXX-NEXT:       MachinePostDominator Tree Construction
; LAXX-NEXT:       Lazy Machine Block Frequency Analysis
; LAXX-NEXT:       Machine Optimization Remark Emitter
; LAXX-NEXT:       Shrink Wrapping analysis
; LAXX-NEXT:       Prologue/Epilogue Insertion & Frame Finalization
; LAXX-NEXT:       Machine Late Instructions Cleanup Pass
; LAXX-NEXT:       Control Flow Optimizer
; LAXX-NEXT:       Lazy Machine Block Frequency Analysis
; LAXX-NEXT:       Tail Duplication
; LAXX-NEXT:       Machine Copy Propagation Pass
; LAXX-NEXT:       Post-RA pseudo instruction expansion pass
; LAXX-NEXT:       MachineDominator Tree Construction
; LAXX-NEXT:       Machine Natural Loop Construction
; LAXX-NEXT:       Post RA top-down list latency scheduler
; LAXX-NEXT:       Analyze Machine Code For Garbage Collection
; LAXX-NEXT:       Machine Block Frequency Analysis
; LAXX-NEXT:       MachinePostDominator Tree Construction
; LAXX-NEXT:       Branch Probability Basic Block Placement
; LAXX-NEXT:       Insert fentry calls
; LAXX-NEXT:       Insert XRay ops
; LAXX-NEXT:       Implement the 'patchable-function' attribute
; LAXX-NEXT:       Branch relaxation pass
; LAXX-NEXT:       Contiguously Lay Out Funclets
; LAXX-NEXT:       StackMap Liveness Analysis
; LAXX-NEXT:       Live DEBUG_VALUE analysis
; LAXX-NEXT:       Machine Sanitizer Binary Metadata
; LAXX-NEXT:       Lazy Machine Block Frequency Analysis
; LAXX-NEXT:       Machine Optimization Remark Emitter
; LAXX-NEXT:       Stack Frame Layout Analysis
; LAXX-NEXT:       LoongArch pseudo instruction expansion pass
; LAXX-NEXT:       LoongArch atomic pseudo instruction expansion pass
; LAXX-NEXT:       Lazy Machine Block Frequency Analysis
; LAXX-NEXT:       Machine Optimization Remark Emitter
; LAXX-NEXT:       LoongArch Assembly Printer
; LAXX-NEXT:       Free MachineFunction
