//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def VectorizeOptions : OptionsStruct;

// LoopIdiomVectorize.cpp
defm : BoolField<"disable-loop-idiom-vectorize-all", "false",
  "Disable Loop Idiom Vectorize Pass">;
defm : BoolField<"disable-loop-idiom-vectorize-bytecmp", "false",
  "Proceed with Loop Idiom Vectorize Pass, but do not convert byte-compare loop(s)">;
defm : BoolField<"disable-loop-idiom-vectorize-find-first-byte", "false",
  "Do not convert find-first-byte loop(s)">;
defm : OptionalValueField<"loop-idiom-vectorize-bytecmp-vf", "unsigned",
  "The vectorization factor for byte-compare patterns">;
defm : OptionalEnumField<"loop-idiom-vectorize-style", "LoopIdiomVectorizeStyle", "masked,predicated",
  ["LoopIdiomVectorizeStyle::Masked", "LoopIdiomVectorizeStyle::Predicated"],
  "The vectorization style for loop idiom transform">;
defm : BoolField<"loop-idiom-vectorize-verify", "false",
  "Verify loops generated Loop Idiom Vectorize Pass">;

// LoopVectorizationLegality.cpp
defm : BoolField<"enable-histogram-loop-vectorization", "false",
  "Enables autovectorization of some loops containing histograms">;
defm : BoolField<"enable-if-conversion", "true",
  "Enable if-conversion during vectorization">;
defm : BoolField<"hints-allow-reordering", "true",
  "Allow enabling loop hints to reorder FP operations during vectorization">;
defm : BoolField<"lv-strided-pointer-ivs", "false",
  "Enable recognition of non-constant strided pointer induction variables">;
defm : EnumField<"scalable-vectorization", "LoopVectorizeHints::ScalableForceKind", "LoopVectorizeHints::SK_Unspecified", "off,preferred,on,always",
  ["LoopVectorizeHints::SK_FixedWidthOnly", "LoopVectorizeHints::SK_PreferScalable", "LoopVectorizeHints::SK_PreferScalable", "LoopVectorizeHints::SK_AlwaysScalable"],
  "Control whether the compiler can use scalable vectors to vectorize a loop">;

// LoopVectorizationPlanner.cpp
defm : BoolField<"force-target-supports-gather-scatter-ops", "false",
  "Assume the target supports gather/scatter operations (used for testing)">;
// Note: This currently only applies to `llvm.masked.load` and
// `llvm.masked.store`. TODO: Extend this to cover other operations as needed.
defm : BoolField<"force-target-supports-masked-memory-ops", "false",
  "Assume the target supports masked memory operations (used for testing)">;
defm : BoolField<"force-target-supports-scalable-vectors", "false",
  "Pretend that scalable vectors are supported, even if the target does not support them. This flag should only be used for testing">;
defm : BoolField<"prefer-inloop-reductions", "false",
  "Prefer in-loop vector reductions, overriding the targets preference">;
defm : ValueField<"scalable-epilogue-vf-cost-scale-factor", "float", "2.0",
  "Scale the cost of scalable epilogue VFs by this factor">;
defm : OptionalBoolField<"vectorizer-consider-reg-pressure",
  "Discard VFs if their register pressure is too high">;
defm : OptionalBoolField<"vectorizer-maximize-bandwidth",
  "Maximize bandwidth when selecting vectorization factor which will be determined by the smallest type in loop">;
defm : BoolField<"vectorizer-maximize-bandwidth-for-vector-calls", "true",
  "Try wider VFs if they enable the use of vector variants">;

// LoopVectorize.cpp
defm : BoolField<"enable-early-exit-vectorization", "true",
  "Enable vectorization of early exit loops with uncountable exits">;
defm : BoolField<"enable-early-exit-vectorization-with-side-effects", "false",
  "Enable vectorization of early exit loops with uncountable exits and side effects">;
defm : BoolField<"enable-epilogue-vectorization", "true",
  "Enable vectorization of epilogue loops">;
defm : BoolField<"enable-ind-var-reg-heur", "true",
  "Count the induction variable only once when interleaving">;
defm : OptionalBoolField<"enable-interleaved-mem-accesses",
  "Enable vectorization on interleaved memory accesses in a loop">;
// Runtime interleave loops for load/store throughput.
defm : BoolField<"enable-loadstore-runtime-interleave", "true",
  "Enable runtime interleaving until load/store ports are saturated">;
// An interleave-group may need masking if it resides in a block that needs
// predication, or in order to mask away gaps.
defm : OptionalBoolField<"enable-masked-interleaved-mem-accesses",
  "Enable vectorization on masked interleaved memory accesses in a loop">;
defm : BoolField<"enable-vplan-native-path", "false",
  "Enable VPlan-native vectorization path with support for outer loop vectorization">;
defm : OptionalEnumField<"epilogue-tail-folding-policy", "TailFoldingPolicyTy", "dont-fold-tail,prefer-fold-tail",
  ["TailFoldingPolicyTy::None", "TailFoldingPolicyTy::PreferFoldTail"],
  "Epilogue-tail-folding preferences over creating an epilogue loop">;
defm : OptionalValueField<"epilogue-vectorization-minimum-VF", "unsigned",
  "Only loops with vectorization factor equal to or larger than the specified value are considered for epilogue vectorization">;
defm : OptionalBoolField<"force-ordered-reductions",
  "Enable the vectorisation of loops with in-order (strict) FP reductions">;
defm : BoolField<"force-partial-aliasing-vectorization", "false",
  "Replace pointer diff checks with alias masks">;
defm : OptionalEnumField<"force-tail-folding-style", "TailFoldingStyle", "none,data,data-without-lane-mask,data-and-control,data-with-evl",
  ["TailFoldingStyle::None", "TailFoldingStyle::Data", "TailFoldingStyle::DataWithoutLaneMask", "TailFoldingStyle::DataAndControlFlow", "TailFoldingStyle::DataWithEVL"],
  "Force the tail folding style">;
defm : OptionalValueField<"force-target-instruction-cost", "unsigned",
  "A flag that overrides the target's expected cost for an instruction to a single constant value. Mostly useful for getting consistent testing">;
defm : OptionalValueField<"force-target-max-scalar-interleave", "unsigned",
  "A flag that overrides the target's max interleave factor for scalar loops">;
defm : OptionalValueField<"force-target-max-vector-interleave", "unsigned",
  "A flag that overrides the target's max interleave factor for vectorized loops">;
defm : OptionalValueField<"force-target-num-scalar-regs", "unsigned",
  "A flag that overrides the target's number of scalar registers">;
defm : OptionalValueField<"force-target-num-vector-regs", "unsigned",
  "A flag that overrides the target's number of vector registers">;
defm : OptionalBoolField<"force-widen-divrem-via-masked-intrinsic",
  "Override cost based masked intrinsic widening for div/rem instructions">;
defm : BoolField<"interleave-loops", "true",
  "Enable loop interleaving in Loop vectorization passes">;
defm : BoolField<"loop-vectorize-with-block-frequency", "true",
  "Enable the use of the block frequency analysis to access PGO heuristics minimizing code growth in cold regions and being more aggressive in hot regions">;
defm : ValueField<"low-trip-count-loop-body-size-limit", "unsigned", "20",
  "Minimum number of instructions to vectorize loops with trip counts below tail folding threshold">;
defm : ValueField<"max-nested-scalar-reduction-interleave", "unsigned", "2",
  "The maximum interleave count to use when interleaving a scalar reduction in a nested loop">;
defm : ValueField<"pragma-vectorize-scev-check-threshold", "unsigned", "128",
  "The maximum number of SCEV checks allowed with a vectorize(enable) pragma">;
defm : BoolField<"prefer-predicated-reduction-select", "false",
  "Prefer predicating a reduction operation over an after loop select">;
defm : ValueField<"small-loop-cost", "unsigned", "20",
  "The cost of a loop that is considered 'small' by the interleaver">;
defm : OptionalEnumField<"tail-folding-policy", "TailFoldingPolicyTy", "dont-fold-tail,prefer-fold-tail,must-fold-tail",
  ["TailFoldingPolicyTy::None", "TailFoldingPolicyTy::PreferFoldTail", "TailFoldingPolicyTy::MustFoldTail"],
  "Tail-folding preferences over creating an epilogue loop">;
defm : BoolField<"vectorize-loops", "true",
  "Run the Loop vectorization passes">;
// The number of stores in a loop that are allowed to need predication.
defm : ValueField<"vectorize-num-stores-pred", "unsigned", "1",
  "Max number of stores to be predicated behind an if">;
// TODO: Move size-based thresholds out of legality checking, make cost based
// decisions instead of hard thresholds.
defm : ValueField<"vectorize-scev-check-threshold", "unsigned", "16",
  "The maximum number of SCEV checks allowed">;
// Loops with a known constant trip count below this number are vectorized only
// if no scalar iteration overheads are incurred.
defm : ValueField<"vectorizer-min-trip-count", "unsigned", "16",
  "Loops with a constant trip count that is smaller than this value are vectorized only if no scalar iteration overheads are incurred">;
// This flag enables the stress testing of the VPlan H-CFG construction in the
// VPlan-native vectorization path. It must be used in conjuction with
// -enable-vplan-native-path. -vplan-verify-hcfg can also be used to enable the
// verification of the H-CFGs built.
defm : BoolField<"vplan-build-outerloop-stress-test", "false",
  "Build VPlan for every supported loop nest in the function and bail out right after the build (stress test the VPlan H-CFG construction in the VPlan-native vectorization path)">;
let Condition = AssertsOnly in {
defm : ListField<"vplan-print-after", "StringRef",
  "Print VPlans after specified VPlan transformations (regexp)">;
defm : BoolField<"vplan-print-after-all", "false",
  "Print VPlans after all VPlan transformations">;
defm : ListField<"vplan-print-before", "StringRef",
  "Print VPlans before specified VPlan transformations (regexp)">;
defm : BoolField<"vplan-print-before-all", "false",
  "Print VPlans before all VPlan transformations">;
defm : BoolField<"vplan-print-vector-region-scope", "false",
  "Limit VPlan printing to vector loop region in `-vplan-print-after*` if the plan has one">;
}
defm : BoolField<"vplan-verify-each", "VPlanVerifyEachDefault",
  "Verify VPlans after VPlan transforms">;

// SandboxVectorizer/Passes/BundleVec.cpp
let Condition = AssertsOnly in
defm : BoolField<"sbvec-always-verify", "false",
  "Helps find bugs by verifying the IR whenever we emit new instructions (*very* expensive)">;
defm : ValueField<"sbvec-stop-at", "unsigned long", "SBVecStopDisabled",
  "Vectorize if the invocation count is < than this. 0 disables vectorization">;
defm : ValueField<"sbvec-stop-bndl", "unsigned long", "SBVecStopDisabled",
  "Vectorize up to this many bundles">;

// SandboxVectorizer/Passes/SeedCollection.cpp
defm : BoolField<"sbvec-allow-non-pow2", "false",
  "Allow non-power-of-2 vectorization">;
defm : ValueField<"sbvec-collect-seeds", "StringRef", "\"stores\"",
  "Collect these seeds. Use empty for none or a comma-separated list of 'stores' and 'loads'">;
defm : ValueField<"sbvec-vec-reg-bits", "unsigned", "0",
  "Override the vector register size in bits, which is otherwise found by querying TTI">;

// SandboxVectorizer/Passes/TransactionAcceptOrRevert.cpp
defm : ValueField<"sbvec-cost-threshold", "int", "0",
  "Vectorization cost threshold">;

// SandboxVectorizer/SandboxVectorizer.cpp
// This option is useful for bisection debugging. For example you may use it to
// figure out which filename is the one causing a miscompile. You can specify a
// regex for the filename like: "/[a-m][^/]*" which will enable any file name
// starting with 'a' to 'm' and disable the rest. If the miscompile goes away,
// then we try "/[n-z][^/]*" for the other half of the range, from 'n' to 'z'.
// If we can reproduce the miscompile then we can keep looking in [n-r] and
// [s-z] and so on, in a binary-search fashion.
//
// Please note that we are using [^/]* and not .* to make sure that we are
// matching the actual filename and not some other directory in the path.
defm : ValueField<"sbvec-allow-files", "StringRef", "\".*\"",
  "Run the vectorizer only on file paths that match any in the list of comma-separated regex's">;
defm : ValueField<"sbvec-passes", "StringRef", "\"*\"",
  "Comma-separated list of vectorizer passes. If not set we run the predefined pipeline">;
defm : BoolField<"sbvec-print-pass-pipeline", "false",
  "Prints the pass pipeline and returns">;

// SandboxVectorizer/SeedCollector.cpp
defm : ValueField<"sbvec-seed-bundle-size-limit", "unsigned", "32",
  "Limit the size of the seed bundle to cap compilation time">;
defm : ValueField<"sbvec-seed-groups-limit", "unsigned", "256",
  "Limit the number of collected seeds groups in a BB to cap compilation time">;

// SandboxVectorizer/VecUtils.cpp
defm : ValueField<"sbvec-max-users-to-consider", "unsigned", "16",
  "Limit the number of a seed's users that getNextUserBundles() will examine as candidates for a matching bundle, to cap compilation time">;

// SLPVectorizer.cpp
// Enables vectorization of copyable elements.
defm : BoolField<"slp-copyable-elements", "true",
  "Try to replace values with the idempotent instructions for better vectorization">;
defm : ValueField<"slp-cost-loop-trip-count", "unsigned", "2",
  "Loop trip count, considered by the cost model during modeling (0=loops are ignored and considered flat code)">;
defm : BoolField<"slp-disable-tree-reorder", "false",
  "Disable tree reordering even if it is profitable. Used for testing only">;
defm : BoolField<"slp-enable-masked-stores", "true",
  "Enable vectorization of non-consecutive stores as a single masked store, when the target supports masked stores">;
defm : BoolField<"slp-enable-strided-stores", "false",
  "Enable SLP trees to be built from strided store chains">;
defm : BoolField<"slp-force-strided-loads", "false",
  "Generate strided loads even if they are not profitable. Used for testing only">;
defm : BoolField<"slp-inst-count-check", "true",
  "Reject vectorization if vector instruction count exceeds scalar instruction count">;
// The maximum depth that the look-ahead score heuristic will explore. The
// higher this value, the higher the compilation time overhead.
defm : ValueField<"slp-max-look-ahead-depth", "int", "2",
  "The maximum look-ahead depth for operand reordering scores">;
defm : OptionalValueField<"slp-max-reg-size", "int",
  "Attempt to vectorize for this register size in bits">;
// The maximum depth that the look-ahead score heuristic will explore when it
// probing among candidates for vectorization tree roots. The higher this value,
// the higher the compilation time overhead but unlike similar limit for
// operands ordering this is less frequently used, hence impact of higher value
// is less noticeable.
defm : ValueField<"slp-max-root-look-ahead-depth", "int", "2",
  "The maximum look-ahead depth for searching best rooting option">;
// Maximum number of runtime alias checks (one per pair of base objects) that
// may guard a single versioned region.
defm : ValueField<"slp-max-runtime-alias-checks", "unsigned", "8",
  "The maximum number of runtime alias checks generated to guard a single SLP-vectorized region">;
defm : ValueField<"slp-max-stride", "unsigned", "8",
  "The maximum stride, considered to be profitable">;
defm : OptionalValueField<"slp-max-vf", "unsigned",
  "Maximum SLP vectorization factor (0=unlimited)">;
defm : OptionalValueField<"slp-min-reg-size", "int",
  "Attempt to vectorize for this register size in bits">;
defm : ValueField<"slp-min-strided-loads", "unsigned", "2",
  "The minimum number of loads, which should be considered strided, if the stride is > 1 or is runtime value">;
defm : ValueField<"slp-min-strided-stores", "unsigned", "2",
  "The minimum number of stores, which should be considered strided, if the stride is > 1 or is runtime value">;
defm : ValueField<"slp-min-tree-size", "unsigned", "3",
  "Only vectorize small trees if they are fully vectorizable">;
defm : BoolField<"slp-non-vectorizables-as-reductions", "false",
  "Use  non-vectorizable instructions as potential reduction roots">;
// Refine the loop-aware cost scaling of gather/buildvector tree entries by
// using the per-lane execution scale of the operand that feeds each lane,
// instead of a single whole-entry scale. This matches the LICM hoisting
// performed by optimizeGatherSequence() at codegen time: lanes whose operands
// are loop-invariant in an inner loop contribute the outer loop's execution
// scale rather than the inner loop's, which avoids over-costing buildvectors
// that bridge values from outer loop nests into an inner loop.
defm : BoolField<"slp-per-lane-gather-scale", "true",
  "Use per-lane execution scale for gather/buildvector tree entries to model LICM-hoistable buildvector sequences">;
defm : ValueField<"slp-phi-vectorization-budget", "unsigned", "1024",
  "Do not vectorize a bundle of PHI nodes if the product of the bundle size and the number of incoming values exceeds this value, to limit the compile time spent on wide PHIs">;
defm : BoolField<"slp-postprocess-stores-operands", "false",
  "Force vectorization of non-vectorizable stores operands">;
// Gather operands of associative single-use binary chains into one node.
defm : BoolField<"slp-reassociate-ops", "true",
  "Gather operands of associative binary chains into one node">;
// The family-realigned seed already groups the vectorizable columns; the
// VLOperands polish on top is quadratic in the column count, so past this many
// columns keep the seed instead.
defm : ValueField<"slp-reassociate-reorder-limit", "unsigned", "32",
  "Max flattened operand columns for which associative-chain reordering runs the full operand reorder">;
defm : ValueField<"slp-recursion-max-depth", "unsigned", "12",
  "Limit the recursion depth when building a vectorizable tree">;
defm : BoolField<"slp-revec", "false",
  "Enable vectorization for wider vector utilization">;
// The runtime checks and the guard branch execute on both the vector and the
// scalar fallback path, so they add overhead to the scalar code.
defm : ValueField<"slp-runtime-alias-checks-max-scalar-cost-percent", "unsigned", "25",
  "Maximum SLP runtime alias check cost, as a percentage of the guarded scalar region cost, before versioning is rejected to avoid pessimizing the scalar fallback path">;
// Limits the size of scheduling regions in a block. It avoid long compile times
// for _very_ large blocks where vector instructions are spread over a wide
// range. This limit is way higher than needed by real-world functions.
defm : ValueField<"slp-schedule-budget", "int", "100000",
  "Limit the size of the SLP scheduling region per block">;
defm : BoolField<"slp-split-alternate-instructions", "true",
  "Improve the code quality by splitting alternate instructions">;
defm : OptionalValueField<"slp-threshold", "int",
  "Only vectorize if you gain more than this number">;
// The scalar loop left by the loop vectorizer runs only a few iterations or the
// ranges its runtime checks found overlapping, where the checks executed on
// each iteration are unlikely to pay off, so they get a tighter bound.
defm : ValueField<"slp-vec-loop-rt-checks-cost-percent", "unsigned", "10",
  "Maximum SLP runtime alias check cost, as a percentage of the guarded scalar region cost, for blocks of loops already vectorized by the loop vectorizer">;
defm : BoolField<"slp-vectorize-hor", "true",
  "Attempt to vectorize horizontal reductions">;
defm : BoolField<"slp-vectorize-hor-store", "false",
  "Attempt to vectorize horizontal reductions feeding into a store">;
defm : BoolField<"slp-vectorize-loop-acc-rdx", "true",
  "Emit loop-carried horizontal reductions as a vector accumulator in the loop plus a single reduction after it">;
defm : BoolField<"slp-vectorize-non-power-of-2", "false",
  "Try to vectorize with non-power-of-2 number of elements">;
defm : BoolField<"slp-vectorize-once-used", "true",
  "Use instructions with the single user as standalone vectorization seeds">;
defm : BoolField<"slp-vectorize-poor-throughput", "true",
  "Use poor-throughput instructions (e.g. fdiv, frem, fsqrt) as standalone vectorization seeds">;
// Enable versioning of a basic block with runtime alias checks.
defm : BoolField<"slp-vectorize-with-runtime-alias-checks", "true",
  "Allow SLP to version a block with runtime alias checks to vectorize trees blocked by may-alias memory dependencies">;
defm : BoolField<"vectorize-slp", "true",
  "Run the SLP vectorization passes">;
defm : BoolField<"view-slp-tree", "false",
  "Display the SLP trees with Graphviz">;

// VectorCombine.cpp
defm : BoolField<"disable-binop-extract-shuffle", "false",
  "Disable binop extract to shuffle transforms">;
defm : BoolField<"disable-vector-combine", "false",
  "Disable all vector combine transforms">;
defm : ValueField<"vector-combine-max-scan-instrs", "unsigned", "30",
  "Max number of instructions to scan for vector combining">;

// VPlan.cpp
defm : BoolField<"vplan-print-in-dot-format", "false",
  "Use dot format instead of plain text when dumping VPlans">;

// VPlanTransforms.cpp
// TODO: Remove this once the partial reduction intrinsics are no worse than
// normal vector operations.
defm : BoolField<"use-partial-reductions-by-default", "false",
  "Use partial reduction intrinsics for all supported unordered reductions">;
