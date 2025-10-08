; UNSUPPORTED:expensive_checks
; RUN:llc -O0 -mtriple=spirv-- -disable-verify -debug-pass=Structure < %s 2>&1 \
; RUN:   | FileCheck -match-full-lines -strict-whitespace -check-prefix=SPIRV-O0 %s
; RUN:llc -O1 -mtriple=spirv-- -disable-verify -debug-pass=Structure < %s 2>&1 \
; RUN:   | FileCheck -match-full-lines -strict-whitespace -check-prefix=SPIRV-Opt %s
; RUN:llc -O2 -mtriple=spirv-- -disable-verify -debug-pass=Structure < %s 2>&1 \
; RUN:   | FileCheck -match-full-lines -strict-whitespace -check-prefix=SPIRV-Opt %s
; RUN:llc -O3 -mtriple=spirv-- -disable-verify -debug-pass=Structure < %s 2>&1 \
; RUN:   | FileCheck -match-full-lines -strict-whitespace -check-prefix=SPIRV-Opt %s
;
; REQUIRES:asserts

; SPIRV-O0:Target Library Information
; SPIRV-O0-NEXT:Target Pass Configuration
; SPIRV-O0-NEXT:Machine Module Information
; SPIRV-O0-NEXT:Target Transform Information
; SPIRV-O0-NEXT:Create Garbage Collector Module Metadata
; SPIRV-O0-NEXT:Assumption Cache Tracker
; SPIRV-O0-NEXT:Profile summary info
; SPIRV-O0-NEXT:Machine Branch Probability Analysis
; SPIRV-O0-NEXT:  ModulePass Manager
; SPIRV-O0-NEXT:    Pre-ISel Intrinsic Lowering
; SPIRV-O0-NEXT:    FunctionPass Manager
; SPIRV-O0-NEXT:      Expand large div/rem
; SPIRV-O0-NEXT:      Expand fp
; SPIRV-O0-NEXT:      Lower Garbage Collection Instructions
; SPIRV-O0-NEXT:      Shadow Stack GC Lowering
; SPIRV-O0-NEXT:      Remove unreachable blocks from the CFG
; SPIRV-O0-NEXT:      Instrument function entry/exit with calls to e.g. mcount() (post inlining)
; SPIRV-O0-NEXT:      Scalarize Masked Memory Intrinsics
; SPIRV-O0-NEXT:      Expand reduction intrinsics
; SPIRV-O0-NEXT:      SPIR-V Regularizer
; SPIRV-O0-NEXT:    SPIRV prepare functions
; SPIRV-O0-NEXT:    FunctionPass Manager
; SPIRV-O0-NEXT:      Lower invoke and unwind, for unwindless code generators
; SPIRV-O0-NEXT:      Remove unreachable blocks from the CFG
; SPIRV-O0-NEXT:      SPIRV strip convergent intrinsics
; SPIRV-O0-NEXT:    Unnamed pass: implement Pass::getPassName()
; SPIRV-O0-NEXT:    SPIRV CBuffer Access
; SPIRV-O0-NEXT:    SPIRV emit intrinsics
; SPIRV-O0-NEXT:    FunctionPass Manager
; SPIRV-O0-NEXT:      SPIRV legalize bitcast pass
; SPIRV-O0-NEXT:      Prepare callbr
; SPIRV-O0-NEXT:      Safe Stack instrumentation pass
; SPIRV-O0-NEXT:      Insert stack protectors
; SPIRV-O0-NEXT:      Analysis containing CSE Info
; SPIRV-O0-NEXT:      IRTranslator
; SPIRV-O0-NEXT:      Analysis for ComputingKnownBits
; SPIRV-O0-NEXT:      MachineDominator Tree Construction
; SPIRV-O0-NEXT:      SPIRVPreLegalizerCombiner
; SPIRV-O0-NEXT:      SPIRV pre legalizer
; SPIRV-O0-NEXT:      Analysis containing CSE Info
; SPIRV-O0-NEXT:      Legalizer
; SPIRV-O0-NEXT:      SPIRV post legalizer
; SPIRV-O0-NEXT:      Analysis for ComputingKnownBits
; SPIRV-O0-NEXT:      Dominator Tree Construction
; SPIRV-O0-NEXT:      Natural Loop Information
; SPIRV-O0-NEXT:      Lazy Branch Probability Analysis
; SPIRV-O0-NEXT:      Lazy Block Frequency Analysis
; SPIRV-O0-NEXT:      InstructionSelect
; SPIRV-O0-NEXT:      ResetMachineFunction
; SPIRV-O0-NEXT:      Finalize ISel and expand pseudo-instructions
; SPIRV-O0-NEXT:      Local Stack Slot Allocation
; SPIRV-O0-NEXT:      Remove Redundant DEBUG_VALUE analysis
; SPIRV-O0-NEXT:      Fixup Statepoint Caller Saved
; SPIRV-O0-NEXT:      Lazy Machine Block Frequency Analysis
; SPIRV-O0-NEXT:      Machine Optimization Remark Emitter
; SPIRV-O0-NEXT:      Prologue/Epilogue Insertion & Frame Finalization
; SPIRV-O0-NEXT:      Post-RA pseudo instruction expansion pass
; SPIRV-O0-NEXT:      Analyze Machine Code For Garbage Collection
; SPIRV-O0-NEXT:      Insert fentry calls
; SPIRV-O0-NEXT:      Insert XRay ops
; SPIRV-O0-NEXT:      Machine Sanitizer Binary Metadata
; SPIRV-O0-NEXT:      Lazy Machine Block Frequency Analysis
; SPIRV-O0-NEXT:      Machine Optimization Remark Emitter
; SPIRV-O0-NEXT:      Stack Frame Layout Analysis
; SPIRV-O0-NEXT:    SPIRV module analysis
; SPIRV-O0-NEXT:    FunctionPass Manager
; SPIRV-O0-NEXT:      Lazy Machine Block Frequency Analysis
; SPIRV-O0-NEXT:      Machine Optimization Remark Emitter
; SPIRV-O0-NEXT:      SPIRV Assembly Printer
; SPIRV-O0-NEXT:      Free MachineFunction

; SPIRV-Opt:Target Library Information
; SPIRV-Opt-NEXT:Target Pass Configuration
; SPIRV-Opt-NEXT:Machine Module Information
; SPIRV-Opt-NEXT:Target Transform Information
; SPIRV-Opt-NEXT:Assumption Cache Tracker
; SPIRV-Opt-NEXT:Type-Based Alias Analysis
; SPIRV-Opt-NEXT:Scoped NoAlias Alias Analysis
; SPIRV-Opt-NEXT:Profile summary info
; SPIRV-Opt-NEXT:Create Garbage Collector Module Metadata
; SPIRV-Opt-NEXT:Machine Branch Probability Analysis
; SPIRV-Opt-NEXT:  ModulePass Manager
; SPIRV-Opt-NEXT:    Pre-ISel Intrinsic Lowering
; SPIRV-Opt-NEXT:    FunctionPass Manager
; SPIRV-Opt-NEXT:      Expand large div/rem
; SPIRV-Opt-NEXT:      Expand fp
; SPIRV-Opt-NEXT:      Dominator Tree Construction
; SPIRV-Opt-NEXT:      Basic Alias Analysis (stateless AA impl)
; SPIRV-Opt-NEXT:      Natural Loop Information
; SPIRV-Opt-NEXT:      Canonicalize natural loops
; SPIRV-Opt-NEXT:      Scalar Evolution Analysis
; SPIRV-Opt-NEXT:      Loop Pass Manager
; SPIRV-Opt-NEXT:        Canonicalize Freeze Instructions in Loops
; SPIRV-Opt-NEXT:        Induction Variable Users
; SPIRV-Opt-NEXT:        Loop Strength Reduction
; SPIRV-Opt-NEXT:      Basic Alias Analysis (stateless AA impl)
; SPIRV-Opt-NEXT:      Function Alias Analysis Results
; SPIRV-Opt-NEXT:      Merge contiguous icmps into a memcmp
; SPIRV-Opt-NEXT:      Natural Loop Information
; SPIRV-Opt-NEXT:      Lazy Branch Probability Analysis
; SPIRV-Opt-NEXT:      Lazy Block Frequency Analysis
; SPIRV-Opt-NEXT:      Expand memcmp() to load/stores
; SPIRV-Opt-NEXT:      Lower Garbage Collection Instructions
; SPIRV-Opt-NEXT:      Shadow Stack GC Lowering
; SPIRV-Opt-NEXT:      Remove unreachable blocks from the CFG
; SPIRV-Opt-NEXT:      Natural Loop Information
; SPIRV-Opt-NEXT:      Post-Dominator Tree Construction
; SPIRV-Opt-NEXT:      Branch Probability Analysis
; SPIRV-Opt-NEXT:      Block Frequency Analysis
; SPIRV-Opt-NEXT:      Constant Hoisting
; SPIRV-Opt-NEXT:      Replace intrinsics with calls to vector library
; SPIRV-Opt-NEXT:      Lazy Branch Probability Analysis
; SPIRV-Opt-NEXT:      Lazy Block Frequency Analysis
; SPIRV-Opt-NEXT:      Optimization Remark Emitter
; SPIRV-Opt-NEXT:      Partially inline calls to library functions
; SPIRV-Opt-NEXT:      Instrument function entry/exit with calls to e.g. mcount() (post inlining)
; SPIRV-Opt-NEXT:      Scalarize Masked Memory Intrinsics
; SPIRV-Opt-NEXT:      Expand reduction intrinsics
; SPIRV-Opt-NEXT:      SPIR-V Regularizer
; SPIRV-Opt-NEXT:    SPIRV prepare functions
; SPIRV-Opt-NEXT:    FunctionPass Manager
; SPIRV-Opt-NEXT:      Dominator Tree Construction
; SPIRV-Opt-NEXT:      Natural Loop Information
; SPIRV-Opt-NEXT:      CodeGen Prepare
; SPIRV-Opt-NEXT:      Lower invoke and unwind, for unwindless code generators
; SPIRV-Opt-NEXT:      Remove unreachable blocks from the CFG
; SPIRV-Opt-NEXT:      SPIRV strip convergent intrinsics
; SPIRV-Opt-NEXT:    Unnamed pass: implement Pass::getPassName()
; SPIRV-Opt-NEXT:    SPIRV CBuffer Access
; SPIRV-Opt-NEXT:    SPIRV emit intrinsics
; SPIRV-Opt-NEXT:    FunctionPass Manager
; SPIRV-Opt-NEXT:      SPIRV legalize bitcast pass
; SPIRV-Opt-NEXT:      Dominator Tree Construction
; SPIRV-Opt-NEXT:      Basic Alias Analysis (stateless AA impl)
; SPIRV-Opt-NEXT:      Function Alias Analysis Results
; SPIRV-Opt-NEXT:      ObjC ARC contraction
; SPIRV-Opt-NEXT:      Prepare callbr
; SPIRV-Opt-NEXT:      Safe Stack instrumentation pass
; SPIRV-Opt-NEXT:      Insert stack protectors
; SPIRV-Opt-NEXT:      Analysis containing CSE Info
; SPIRV-Opt-NEXT:      Natural Loop Information
; SPIRV-Opt-NEXT:      Post-Dominator Tree Construction
; SPIRV-Opt-NEXT:      Branch Probability Analysis
; SPIRV-Opt-NEXT:      Basic Alias Analysis (stateless AA impl)
; SPIRV-Opt-NEXT:      Function Alias Analysis Results
; SPIRV-Opt-NEXT:      IRTranslator
; SPIRV-Opt-NEXT:      Analysis for ComputingKnownBits
; SPIRV-Opt-NEXT:      MachineDominator Tree Construction
; SPIRV-Opt-NEXT:      SPIRVPreLegalizerCombiner
; SPIRV-Opt-NEXT:      SPIRV pre legalizer
; SPIRV-Opt-NEXT:      Analysis containing CSE Info
; SPIRV-Opt-NEXT:      Legalizer
; SPIRV-Opt-NEXT:      SPIRV post legalizer
; SPIRV-Opt-NEXT:      Analysis for ComputingKnownBits
; SPIRV-Opt-NEXT:      Lazy Branch Probability Analysis
; SPIRV-Opt-NEXT:      Lazy Block Frequency Analysis
; SPIRV-Opt-NEXT:      InstructionSelect
; SPIRV-Opt-NEXT:      ResetMachineFunction
; SPIRV-Opt-NEXT:      Finalize ISel and expand pseudo-instructions
; SPIRV-Opt-NEXT:      Lazy Machine Block Frequency Analysis
; SPIRV-Opt-NEXT:      Early Tail Duplication
; SPIRV-Opt-NEXT:      Optimize machine instruction PHIs
; SPIRV-Opt-NEXT:      Slot index numbering
; SPIRV-Opt-NEXT:      Merge disjoint stack slots
; SPIRV-Opt-NEXT:      Local Stack Slot Allocation
; SPIRV-Opt-NEXT:      Remove dead machine instructions
; SPIRV-Opt-NEXT:      MachineDominator Tree Construction
; SPIRV-Opt-NEXT:      Machine Natural Loop Construction
; SPIRV-Opt-NEXT:      Machine Block Frequency Analysis
; SPIRV-Opt-NEXT:      Early Machine Loop Invariant Code Motion
; SPIRV-Opt-NEXT:      MachineDominator Tree Construction
; SPIRV-Opt-NEXT:      Machine Block Frequency Analysis
; SPIRV-Opt-NEXT:      Machine Common Subexpression Elimination
; SPIRV-Opt-NEXT:      MachinePostDominator Tree Construction
; SPIRV-Opt-NEXT:      Machine Cycle Info Analysis
; SPIRV-Opt-NEXT:      Machine code sinking
; SPIRV-Opt-NEXT:      Peephole Optimizations
; SPIRV-Opt-NEXT:      Remove dead machine instructions
; SPIRV-Opt-NEXT:      Remove Redundant DEBUG_VALUE analysis
; SPIRV-Opt-NEXT:      Fixup Statepoint Caller Saved
; SPIRV-Opt-NEXT:      Lazy Machine Block Frequency Analysis
; SPIRV-Opt-NEXT:      Machine Optimization Remark Emitter
; SPIRV-Opt-NEXT:      Prologue/Epilogue Insertion & Frame Finalization
; SPIRV-Opt-NEXT:      Tail Duplication
; SPIRV-Opt-NEXT:      Post-RA pseudo instruction expansion pass
; SPIRV-Opt-NEXT:      Analyze Machine Code For Garbage Collection
; SPIRV-Opt-NEXT:      Insert fentry calls
; SPIRV-Opt-NEXT:      Insert XRay ops
; SPIRV-Opt-NEXT:      Machine Sanitizer Binary Metadata
; SPIRV-Opt-NEXT:      Lazy Machine Block Frequency Analysis
; SPIRV-Opt-NEXT:      Machine Optimization Remark Emitter
; SPIRV-Opt-NEXT:      Stack Frame Layout Analysis
; SPIRV-Opt-NEXT:    SPIRV module analysis
; SPIRV-Opt-NEXT:    FunctionPass Manager
; SPIRV-Opt-NEXT:      Lazy Machine Block Frequency Analysis
; SPIRV-Opt-NEXT:      Machine Optimization Remark Emitter
; SPIRV-Opt-NEXT:      SPIRV Assembly Printer
; SPIRV-Opt-NEXT:      Free MachineFunction

define void @empty() {
  ret void
}
