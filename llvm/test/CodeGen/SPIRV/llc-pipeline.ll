; RUN: llc -mtriple=spirv-unknown-unknown -debug-pass=Structure < %s -o /dev/null 2>&1 | \
; RUN: grep -v "Verify generated machine code" | FileCheck %s

; REQUIRES: asserts


; CHECK-LABEL: Pass Arguments:
; CHECK-NEXT: Target Library Information
; CHECK-NEXT: Target Pass Configuration
; CHECK-NEXT: Machine Module Information
; CHECK-NEXT: Target Transform Information
; CHECK-NEXT: Type-Based Alias Analysis
; CHECK-NEXT: Scoped NoAlias Alias Analysis
; CHECK-NEXT: Assumption Cache Tracker
; CHECK-NEXT: Profile summary info
; CHECK-NEXT: Create Garbage Collector Module Metadata
; CHECK-NEXT: Machine Branch Probability Analysis
; CHECK-NEXT:   ModulePass Manager
; CHECK-NEXT:     Pre-ISel Intrinsic Lowering
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Expand large div/rem
; CHECK-NEXT:       Expand large fp convert
; CHECK-NEXT:       Module Verifier
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       Canonicalize natural loops
; CHECK-NEXT:       Scalar Evolution Analysis
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
; CHECK-NEXT:       Remove unreachable blocks from the CFG
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       Post-Dominator Tree Construction
; CHECK-NEXT:       Branch Probability Analysis
; CHECK-NEXT:       Block Frequency Analysis
; CHECK-NEXT:       Constant Hoisting
; CHECK-NEXT:       Replace intrinsics with calls to vector library
; CHECK-NEXT:       Partially inline calls to library functions
; CHECK-NEXT:       Instrument function entry/exit with calls to e.g. mcount() (post inlining)
; CHECK-NEXT:       Scalarize Masked Memory Intrinsics
; CHECK-NEXT:       Expand reduction intrinsics
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       Canonicalize natural loops
; CHECK-NEXT:       Unnamed pass: implement Pass::getPassName()
; CHECK-NEXT:       SPIRV convergence regions analysis
; CHECK-NEXT:       SPIRV split region exit blocks
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       structurize SPIRV
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Promote Memory to Register
; CHECK-NEXT:       SPIR-V Regularizer
; CHECK-NEXT:     SPIRV prepare functions
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       SPIRV strip convergent intrinsics
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       CodeGen Prepare
; CHECK-NEXT:       Lower invoke and unwind, for unwindless code generators
; CHECK-NEXT:       Remove unreachable blocks from the CFG
; CHECK-NEXT:     SPIRV emit intrinsics
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       ObjC ARC contraction
; CHECK-NEXT:       Prepare callbr
; CHECK-NEXT:       Safe Stack instrumentation pass
; CHECK-NEXT:       Insert stack protectors
; CHECK-NEXT:       Module Verifier
; CHECK-NEXT:       Analysis containing CSE Info
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       Post-Dominator Tree Construction
; CHECK-NEXT:       Branch Probability Analysis
; CHECK-NEXT:       Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       IRTranslator
; CHECK-NEXT:       Analysis for ComputingKnownBits
; CHECK-NEXT:       MachineDominator Tree Construction
; CHECK-NEXT:       Analysis containing CSE Info
; CHECK-NEXT:       SPIRVPreLegalizerCombiner
; CHECK-NEXT:       SPIRV pre legalizer
; CHECK-NEXT:       Legalizer
; CHECK-NEXT:       SPIRV post legalizer
; CHECK-NEXT:       Analysis for ComputingKnownBits
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       InstructionSelect
; CHECK-NEXT:       ResetMachineFunction
; CHECK-NEXT:       Finalize ISel and expand pseudo-instructions
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Early Tail Duplication
; CHECK-NEXT:       Optimize machine instruction PHIs
; CHECK-NEXT:       Slot index numbering
; CHECK-NEXT:       Merge disjoint stack slots
; CHECK-NEXT:       Local Stack Slot Allocation
; CHECK-NEXT:       Remove dead machine instructions
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
; CHECK-NEXT:       Remove Redundant DEBUG_VALUE analysis
; CHECK-NEXT:       Fixup Statepoint Caller Saved
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Machine Optimization Remark Emitter
; CHECK-NEXT:       Prologue/Epilogue Insertion & Frame Finalization
; CHECK-NEXT:       Tail Duplication
; CHECK-NEXT:       Post-RA pseudo instruction expansion pass
; CHECK-NEXT:       Analyze Machine Code For Garbage Collection
; CHECK-NEXT:       Insert fentry calls
; CHECK-NEXT:       Insert XRay ops
; CHECK-NEXT:       Machine Sanitizer Binary Metadata
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Machine Optimization Remark Emitter
; CHECK-NEXT:       Stack Frame Layout Analysis
; CHECK-NEXT:     SPIRV module analysis
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Machine Optimization Remark Emitter
; CHECK-NEXT:       SPIRV Assembly Printer
; CHECK-NEXT:       Free MachineFunction
