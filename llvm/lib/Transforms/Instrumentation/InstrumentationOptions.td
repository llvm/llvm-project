//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def InstrumentationOptions : OptionsStruct;

// AddressSanitizer.cpp
defm : BoolField<"asan-always-slow-path", "false",
  "Use instrumentation with slow path for all accesses">;
defm : OptionalEnumField<"asan-constructor-kind", "AsanCtorKind",
  "none,global",
  ["AsanCtorKind::None", "AsanCtorKind::Global"],
  "Sets the ASan constructor kind">;
defm : ValueField<"asan-debug-func", "StringRef", "",
  "Do not instrument the function with this name">;
defm : ValueField<"asan-debug-max", "int", "-1",
  "With -asan-debug-min, instrument only the memory accesses whose index within the function is in [min, max]">;
defm : ValueField<"asan-debug-min", "int", "-1",
  "With -asan-debug-max, instrument only the memory accesses whose index within the function is in [min, max]">;
defm : BoolField<"asan-debug-stack", "false",
  "Print each function after stack instrumentation (with -debug-only=asan)">;
defm : OptionalEnumField<"asan-destructor-kind", "AsanDtorKind",
  "none,global",
  ["AsanDtorKind::None", "AsanDtorKind::Global"],
  "Sets the ASan destructor kind. The default is to use the value provided to the pass constructor">;
defm : BoolField<"asan-detect-invalid-pointer-cmp", "false",
  "Instrument <, <=, >, >= with pointer operands">;
defm : BoolField<"asan-detect-invalid-pointer-pair", "false",
  "Instrument <, <=, >, >=, - with pointer operands">;
defm : BoolField<"asan-detect-invalid-pointer-sub", "false",
  "Instrument - operations with pointer operands">;
defm : BoolField<"asan-force-dynamic-shadow", "false",
  "Load shadow address into a local variable for each function">;
defm : ValueField<"asan-force-experiment", "uint32_t", "0",
  "Force optimization experiment (for testing)">;
// This flag may need to be replaced with -f[no]asan-globals.
defm : BoolField<"asan-globals", "true",
  "Handle global objects">;
defm : BoolField<"asan-globals-live-support", "true",
  "Use linker features to support dead code stripping of globals">;
defm : OptionalBoolField<"asan-guard-against-version-mismatch",
  "Guard against compiler/runtime version mismatch">;
defm : BoolField<"asan-initialization-order", "true",
  "Handle C++ initializer order">;
defm : ListField<"asan-instrument-address-spaces", "unsigned",
  "Only instrument variables in the specified address spaces">;
defm : BoolField<"asan-instrument-atomics", "true",
  "Instrument atomic instructions (rmw, cmpxchg)">;
defm : BoolField<"asan-instrument-byval", "true",
  "Instrument byval call arguments">;
defm : BoolField<"asan-instrument-dynamic-allocas", "true",
  "Instrument dynamic allocas">;
// This flag may need to be replaced with -f[no-]asan-reads.
defm : BoolField<"asan-instrument-reads", "true",
  "Instrument read instructions">;
defm : BoolField<"asan-instrument-writes", "true",
  "Instrument write instructions">;
defm : OptionalValueField<"asan-instrumentation-with-call-threshold", "int",
  "If the function being instrumented contains more than this number of memory accesses, use callbacks instead of inline checks (-1 means never use callbacks)">;
defm : OptionalBoolField<"asan-kernel",
  "Enable KernelAddressSanitizer instrumentation">;
defm : BoolField<"asan-kernel-mem-intrinsic-prefix", "false",
  "Use prefix for memory intrinsics in KASAN mode">;
// The shadow mapping looks like
//    Shadow = (Mem >> scale) + offset
defm : OptionalValueField<"asan-mapping-offset", "uint64_t",
  "Offset of asan shadow mapping [EXPERIMENTAL]">;
defm : OptionalValueField<"asan-mapping-scale", "int",
  "Scale of asan shadow mapping">;
defm : OptionalValueField<"asan-max-inline-poisoning-size", "uint32_t",
  "Inline shadow poisoning for blocks up to the given size in bytes">;
// This flag limits the number of instructions to be instrumented in any given
// BB. Normally, this should be set to unlimited (INT_MAX), but due to
// http://llvm.org/bugs/show_bug.cgi?id=12652 we temporary set it to 10000.
defm : ValueField<"asan-max-ins-per-bb", "int", "10000",
  "Maximal number of instructions to instrument in any given BB">;
defm : ValueField<"asan-memory-access-callback-prefix", "StringRef", "\"__asan_\"",
  "Prefix for memory access callbacks">;
defm : BoolField<"asan-opt", "true",
  "Enable -asan-opt-globals, -asan-opt-same-temp, and -asan-opt-stack">;
defm : BoolField<"asan-opt-globals", "true",
  "Don't instrument scalar globals">;
defm : BoolField<"asan-opt-same-temp", "true",
  "Instrument the same temp just once">;
defm : BoolField<"asan-opt-stack", "false",
  "Don't instrument scalar stack variables">;
defm : BoolField<"asan-optimize-callbacks", "false",
  "Emit callback checks as llvm.asan.check.memaccess intrinsics">;
defm : ValueField<"asan-realign-stack", "unsigned", "32",
  "Realign stack to the value of this flag (power of two)">;
defm : OptionalBoolField<"asan-recover",
  "Enable recovery mode (continue-after-error)">;
defm : BoolField<"asan-redzone-byval-args", "true",
  "Create redzones for byval arguments (extra copy required)">;
defm : ValueField<"asan-shadow-addr-space", "int", "0",
  "Address space for pointers to the shadow map">;
defm : BoolField<"asan-skip-promotable-allocas", "true",
  "Do not instrument promotable allocas">;
// This flag may need to be replaced with -f[no]asan-stack.
defm : BoolField<"asan-stack", "true",
  "Handle stack memory">;
defm : BoolField<"asan-stack-dynamic-alloca", "true",
  "Use dynamic alloca to represent stack variables">;
defm : OptionalEnumField<"asan-use-after-return", "AsanDetectStackUseAfterReturnMode",
  "never,runtime,always",
  ["AsanDetectStackUseAfterReturnMode::Never", "AsanDetectStackUseAfterReturnMode::Runtime", "AsanDetectStackUseAfterReturnMode::Always"],
  "Sets the mode of detection for stack-use-after-return">;
defm : BoolField<"asan-use-after-scope", "false",
  "Check stack-use-after-scope">;
defm : OptionalBoolField<"asan-use-odr-indicator",
  "Use odr indicators to improve ODR reporting">;
defm : OptionalBoolField<"asan-use-private-alias",
  "Use private aliases for global variables">;
defm : BoolField<"asan-use-stack-safety", "true",
  "Use Stack Safety analysis results">;
// This is on by default even though there is a bug in gold:
// https://sourceware.org/bugzilla/show_bug.cgi?id=19002
defm : BoolField<"asan-with-comdat", "true",
  "Place ASan constructors in comdat sections">;
defm : BoolField<"asan-with-ifunc", "true",
  "Access dynamic shadow through an ifunc global on platforms that support this">;
defm : BoolField<"asan-with-ifunc-suppress-remat", "true",
  "Suppress rematerialization of dynamic shadow address by passing it through inline asm in prologue">;

// AllocToken.cpp
// C++ defines ::operator new (and variants) as replaceable (vs. standard
// library versions), which are nobuiltin, and are therefore not covered by
// isAllocationFn(). Cover by default, as users of AllocToken are already
// required to provide token-aware allocation functions (no defaults).
defm : BoolField<"alloc-token-cover-replaceable-new", "true",
  "Cover replaceable operator new">;
// Instrument libcalls only by default - compatible allocators only need to take
// care of providing standard allocation functions. With extended coverage, also
// instrument non-libcall allocation function calls with !alloc_token metadata.
defm : OptionalBoolField<"alloc-token-extended",
  "Extend coverage to custom allocation functions">;
defm : ValueField<"alloc-token-fallback", "uint64_t", "0",
  "The default fallback token where none could be determined">;
defm : OptionalBoolField<"alloc-token-fast-abi",
  "The token ID is encoded in the function name">;
defm : OptionalValueField<"alloc-token-max", "uint64_t",
  "Maximum number of tokens (0 = target SIZE_MAX)">;
defm : ValueField<"alloc-token-prefix", "StringRef", "\"__alloc_token_\"",
  "The allocation function prefix">;

// BoundsChecking.cpp
defm : BoolField<"bounds-checking-single-trap", "false",
  "Use one trap block per function">;

// ControlHeightReduction.cpp
defm : ValueField<"chr-bias-threshold", "double", "0.99",
  "CHR considers a branch bias greater than this ratio as biased">;
defm : ValueField<"chr-dup-threshold", "unsigned", "3",
  "Max number of duplications by CHR for a region">;
defm : ValueField<"chr-function-list", "StringRef", "",
  "Specify file to retrieve the list of functions to apply CHR to">;
defm : ValueField<"chr-merge-threshold", "unsigned", "2",
  "CHR merges a group of N branches/selects where N >= this value">;
defm : ValueField<"chr-module-list", "StringRef", "",
  "Specify file to retrieve the list of modules to apply CHR to">;
defm : BoolField<"disable-chr", "false",
  "Disable CHR for all functions">;
defm : BoolField<"force-chr", "false",
  "Apply CHR for all functions">;

// DataFlowSanitizer.cpp
defm : ListField<"dfsan-abilist", "StringRef",
  "File listing native ABI functions and how the pass treats them">;
defm : BoolField<"dfsan-add-global-name-suffix", "true",
  "Whether to add .dfsan suffix to global names">;
// Controls whether the pass propagates labels of offsets in GEP instructions.
defm : BoolField<"dfsan-combine-offset-labels-on-gep", "true",
  "Combine the label of the offset with the label of the pointer when doing pointer arithmetic">;
// Controls whether the pass includes or ignores the labels of pointers in load
// instructions.
defm : BoolField<"dfsan-combine-pointer-labels-on-load", "true",
  "Combine the label of the pointer with the label of the data when loading from memory">;
// Controls whether the pass includes or ignores the labels of pointers in
// stores instructions.
defm : BoolField<"dfsan-combine-pointer-labels-on-store", "false",
  "Combine the label of the pointer with the label of the data when storing in memory">;
defm : ListField<"dfsan-combine-taint-lookup-table", "StringRef",
  "When dfsan-combine-offset-labels-on-gep and/or dfsan-combine-pointer-labels-on-load are false, this flag can be used to re-enable combining offset and/or pointer taint when loading specific constant global variables (i.e. lookup tables)">;
// Experimental feature that inserts callbacks for conditionals, including:
// conditional branch, switch, select. This must be true for
// dfsan_set_conditional_callback() to have effect.
defm : BoolField<"dfsan-conditional-callbacks", "false",
  "Insert calls to callback functions on conditionals">;
defm : BoolField<"dfsan-debug-nonzero-labels", "false",
  "Insert calls to __dfsan_nonzero_label on observing a parameter, load or return with a nonzero label">;
// Experimental feature that inserts callbacks for certain data events.
// Currently callbacks are only inserted for loads, stores, memory transfers
// (i.e. memcpy and memmove), and comparisons.
//
// If this flag is set to true, the user must provide definitions for the
// following callback functions:
//   void __dfsan_load_callback(dfsan_label Label, void* addr);
//   void __dfsan_store_callback(dfsan_label Label, void* addr);
//   void __dfsan_mem_transfer_callback(dfsan_label *Start, size_t Len);
//   void __dfsan_cmp_callback(dfsan_label CombinedLabel);
defm : BoolField<"dfsan-event-callbacks", "false",
  "Insert calls to __dfsan_*_callback functions on data events">;
defm : BoolField<"dfsan-ignore-personality-routine", "false",
  "If a personality routine is marked uninstrumented from the ABI list, do not create a wrapper for it">;
// TODO: This default value follows MSan. DFSan may use a different value.
defm : ValueField<"dfsan-instrument-with-call-threshold", "int", "3500",
  "If the function being instrumented requires more than this number of origin stores, use callbacks instead of inline checks (-1 means never use callbacks)">;
// The -dfsan-preserve-alignment flag controls whether this pass assumes that
// alignment requirements provided by the input IR are correct. For example, if
// the input IR contains a load with alignment 8, this flag will cause the
// shadow load to have alignment 16. This flag is disabled by default as we have
// unfortunately encountered too much code (including Clang itself; see PR14291)
// which performs misaligned access.
defm : BoolField<"dfsan-preserve-alignment", "false",
  "Respect alignment requirements provided by input IR">;
// Experimental feature that inserts callbacks for data reaching a function,
// either via function arguments and loads. This must be true for
// dfsan_set_reaches_function_callback() to have effect.
defm : BoolField<"dfsan-reaches-function-callbacks", "false",
  "Insert calls to callback functions on data reaching a function">;
// Controls how to track origins.
// * 0: do not track origins.
// * 1: track origins at memory store operations.
// * 2: track origins at memory load and store operations.
//      TODO: track callsites.
defm : ValueField<"dfsan-track-origins", "int", "0",
  "Track origins of labels">;
// Controls whether the pass tracks the control flow of select instructions.
defm : BoolField<"dfsan-track-select-control-flow", "true",
  "Propagate labels from condition values of select instructions to results">;

// GCOVProfiling.cpp
defm : ValueField<"default-gcov-version", "StringRef", "\"0000\"", "">;
defm : BoolField<"gcov-atomic-counter", "false",
  "Make counter updates atomic">;

// HWAddressSanitizer.cpp
defm : BoolField<"hwasan-all-globals", "false",
  "Instrument globals, even those within user-defined sections. Warning: This may break existing code which walks globals via linker-generated symbols, expects certain globals to be contiguous with each other, or makes other assumptions which are invalidated by HWASan instrumentation">;
// Enabled from clang by "-fsanitize-hwaddress-experimental-aliasing".
defm : BoolField<"hwasan-experimental-use-page-aliases", "false",
  "Use page aliasing in HWASan">;
defm : BoolField<"hwasan-generate-tags-with-calls", "false",
  "Generate new tags with runtime library calls">;
defm : OptionalBoolField<"hwasan-globals",
  "Instrument globals">;
defm : OptionalBoolField<"hwasan-inline-all-checks",
  "Inline all checks">;
defm : OptionalBoolField<"hwasan-inline-fast-path-checks",
  "Inline all checks">;
defm : BoolField<"hwasan-instrument-atomics", "true",
  "Instrument atomic instructions (rmw, cmpxchg)">;
defm : BoolField<"hwasan-instrument-byval", "true",
  "Instrument byval arguments">;
defm : OptionalBoolField<"hwasan-instrument-landing-pads",
  "Instrument landing pads">;
defm : BoolField<"hwasan-instrument-mem-intrinsics", "true",
  "Instrument memory intrinsics">;
defm : OptionalBoolField<"hwasan-instrument-personality-functions",
  "Instrument personality functions">;
defm : BoolField<"hwasan-instrument-reads", "true",
  "Instrument read instructions">;
defm : BoolField<"hwasan-instrument-stack", "true",
  "Instrument stack (allocas)">;
defm : OptionalBoolField<"hwasan-instrument-with-calls",
  "Instrument reads and writes with callbacks">;
defm : BoolField<"hwasan-instrument-writes", "true",
  "Instrument write instructions">;
defm : OptionalBoolField<"hwasan-kernel",
  "Enable KernelHWAddressSanitizer instrumentation">;
defm : BoolField<"hwasan-kernel-mem-intrinsic-prefix", "false",
  "Use prefix for memory intrinsics in KASAN mode">;
defm : OptionalValueField<"hwasan-match-all-tag", "int",
  "Don't report bad accesses via pointers with this tag">;
defm : ValueField<"hwasan-max-lifetimes-for-alloca", "size_t", "3",
  "How many lifetime ends to handle for a single alloca">;
defm : ValueField<"hwasan-memory-access-callback-prefix", "StringRef", "\"__hwasan_\"",
  "Prefix for memory access callbacks">;
defm : OptionalValueField<"hwasan-percentile-cutoff-hot", "int",
  "Hot percentile cutoff">;
defm : OptionalValueField<"hwasan-random-rate", "float",
  "Probability value in the range [0.0, 1.0] to keep instrumentation of a function. Note: instrumentation can be skipped randomly OR because of the hot percentile cutoff, if both are supplied">;
defm : EnumField<"hwasan-record-stack-history", "RecordStackHistoryMode", "RecordStackHistoryMode::instr",
  "none,instr,libcall",
  ["RecordStackHistoryMode::none", "RecordStackHistoryMode::instr", "RecordStackHistoryMode::libcall"],
  "Record stack frames with tagged allocations in a thread-local ring buffer">;
defm : OptionalBoolField<"hwasan-recover",
  "Enable recovery mode (continue-after-error)">;
defm : BoolField<"hwasan-static-linking", "false",
  "Don't use .note.hwasan.globals section to instrument globals from loadable libraries. Note: in static binaries, the global variables section can be accessed directly via linker-provided __start_hwasan_globals and __stop_hwasan_globals symbols">;
defm : BoolField<"hwasan-strict-use-after-scope", "true",
  "For complicated lifetimes, tag both on end and return">;
defm : ValueField<"hwasan-tag-bits", "uint64_t", "0",
  "Restrict tag to at most N bits. Needs to be > 4">;
defm : BoolField<"hwasan-use-after-scope", "true",
  "Detect use after scope within function">;
defm : OptionalBoolField<"hwasan-use-short-granules",
  "Use short granules in allocas and outlined checks">;
defm : OptionalBoolField<"hwasan-use-stack-safety",
  "Use Stack Safety analysis results">;
defm : OptionalBoolField<"hwasan-with-frame-record",
  "Use ring buffer for stack allocations">;

// IndirectCallPromotion.cpp
// Command line option to disable indirect-call promotion with the default as
// false. This is for debug purpose.
defm : BoolField<"disable-icp", "false",
  "Disable indirect call promotion">;
defm : ValueField<"hot-func-cutoff-for-icp", "int", "-1",
  "A count is hot for indirect call promotion if it exceeds the minimum count to reach this percentile of total counts.Note that this percentile is specified as percentile * 10000 = HotFuncCutoffForICP.Default value -1 means that if the flag is unspecified then the value of ProfileSummaryCutoffHot will be used instead">;
// If one target cannot be ICP'd, proceed with the remaining targets instead of
// exiting the callsite.
defm : BoolField<"icp-allow-candidate-skip", "false",
  "Continue with the remaining targets instead of exiting when failing in a candidate">;
// ICP the candidate function even when only a declaration is present.
defm : BoolField<"icp-allow-decls", "false",
  "Promote the target candidate even when the definition  is not available">;
// ICP hot candidate functions only. When setting to false, non-cold functions
// (warm functions) can also be promoted.
defm : BoolField<"icp-allow-hot-only", "true",
  "Promote the target candidate only if it is a hot function. Otherwise, warm functions can also be promoted">;
// If the option is set to true, only call instructions will be considered for
// transformation -- invoke instructions will be ignored.
defm : BoolField<"icp-call-only", "false",
  "Run indirect-call promotion for call instructions only">;
// If -icp-csskip is non zero, the first -icp-csskip callsites will be skipped.
// For debug use only.
defm : ValueField<"icp-csskip", "unsigned", "0",
  "Skip Callsite up to this number for this compilation">;
// Set the cutoff value for the promotion. If the value is other than 0, we stop
// the transformation once the total number of promotions equals the cutoff
// value. For debug use only.
defm : ValueField<"icp-cutoff", "unsigned", "0",
  "Max number of promotions for this compilation">;
// Dump the function level IR if the transformation happened in this function.
// For debug use only.
defm : BoolField<"icp-dumpafter", "false",
  "Dump IR after transformation happens">;
defm : ListField<"icp-ignored-base-types", "StringRef",
  "A list of mangled vtable type info names. Classes specified by the type info names and their derived ones will not be vtable-ICP'ed. Useful when the profiled types and actual types in the optimized binary could be different due to profiling limitations. Type info names are those string literals used in LLVM type metadata">;
// If the option is set to true, only invoke instructions will be considered for
// transformation -- call instructions will be ignored.
defm : BoolField<"icp-invoke-only", "false",
  "Run indirect-call promotion for invoke instruction only">;
// Set if the pass is called in LTO optimization. The difference for LTO mode is
// the pass won't prefix the source module name to the internal linkage symbols.
defm : BoolField<"icp-lto", "false",
  "Run indirect-call promotion in LTO mode">;
// Although comparing vtables can save a vtable load, we may need to compare
// vtable pointer with multiple vtable address points due to class inheritance.
// Comparing with multiple vtables inserts additional instructions on hot code
// path, and doing so for an earlier candidate delays the comparisons for later
// candidates. For the last candidate, only the fallback path is affected. We
// allow multiple vtable comparison for the last function candidate and use the
// option below to cap the number of vtables.
defm : ValueField<"icp-max-num-vtable-last-candidate", "int", "1",
  "The maximum number of vtable for the last candidate">;
// Set if the pass is called in SamplePGO mode. The difference for SamplePGO
// mode is it will add prof metadatato the created direct call.
defm : BoolField<"icp-samplepgo", "false",
  "Run indirect-call promotion in SamplePGO mode">;
// Indirect call promotion pass will fall back to function-based comparison if
// vtable-count / function-count is smaller than this threshold.
defm : ValueField<"icp-vtable-percentage-threshold", "float", "0.995",
  "The percentage threshold of vtable-count / function-count for cost-benefit analysis">;

// InstrProfiling.cpp
defm : BoolField<"atomic-counter-update-promoted", "false",
  "Do counter update using atomic fetch add  for promoted counters only">;
defm : BoolField<"atomic-first-counter", "false",
  "Use atomic fetch add for first counter in a function (usually the entry counter)">;
defm : BoolField<"conditional-counter-update", "false",
  "Do conditional counter updates in single byte counters mode)">;
// If the option is not specified, the default behavior about whether counter
// promotion is done depends on how instrumentation lowering pipeline is setup,
// i.e., the default value of true of this option does not mean the promotion
// will be done by default. Explicitly setting this option can override the
// default behavior.
defm : OptionalBoolField<"do-counter-promotion",
  "Do counter register promotion">;
defm : BoolField<"hash-based-counter-split", "true",
  "Rename counter variable of a comdat function based on cfg hash">;
defm : BoolField<"instrprof-atomic-counter-update-all", "false",
  "Make all profile counter updates atomic (for testing only)">;
defm : BoolField<"iterative-counter-promotion", "true",
  "Allow counter promotion across the whole loop nest">;
// A debug option.
defm : ValueField<"max-counter-promotions", "int", "-1",
  "Max number of allowed counter promotions">;
defm : ValueField<"max-counter-promotions-per-loop", "unsigned", "20",
  "Max number counter promotions per loop to avoid increasing register pressure too much">;
defm : ValueField<"offload-pgo-sampling", "unsigned", "3",
  "Log2 of the sampling period for offload PGO instrumentation. Only 1 in every 2^N blocks is instrumented. 0 = all blocks, 1 = 50%, 2 = 25%, 3 = 12.5%. Higher values reduce overhead at the cost of sparser profiles">;
defm : EnumField<"profile-correlate", "InstrProfCorrelator::ProfCorrelatorKind", "InstrProfCorrelator::NONE",
  "debug-info,binary",
  ["InstrProfCorrelator::DEBUG_INFO", "InstrProfCorrelator::BINARY"],
  "Use debug info or binary file to correlate profiles">;
defm : OptionalBoolField<"runtime-counter-relocation",
  "Enable relocating counters at runtime">;
defm : ValueField<"sampled-instr-burst-duration", "unsigned", "200",
  "Set the profile instrumentation burst duration, which can range from 1 to the value of 'sampled-instr-period' (0 is invalid). This number of samples will be recorded for each 'sampled-instr-period' count update. Setting to 1 enables simple sampling, in which case it is recommended to set 'sampled-instr-period' to a prime number">;
defm : ValueField<"sampled-instr-period", "unsigned", "65536",
  "Set the profile instrumentation sample period. A sample period of 0 is invalid. For each sample period, a fixed number of consecutive samples will be recorded. The number is controlled by 'sampled-instr-burst-duration' flag. The default sample period of 65536 is optimized for generating efficient code that leverages unsigned short integer wrapping in overflow, but this is disabled under simple sampling (burst duration = 1)">;
defm : OptionalBoolField<"sampled-instrumentation",
  "Do PGO instrumentation sampling">;
defm : BoolField<"skip-ret-exit-block", "true",
  "Suppress counter promotion if exit blocks contain ret">;
defm : ValueField<"speculative-counter-promotion-max-exiting", "unsigned", "3",
  "The max number of exiting blocks of a loop to allow  speculative counter promotion">;
defm : BoolField<"speculative-counter-promotion-to-loop", "false",
  "When the option is false, if the target block is in a loop, the promotion will be disallowed unless the promoted counter  update can be further/iteratively promoted into an acyclic  region">;
defm : BoolField<"verify-atomic-counter-promoted", "false",
  "Check that all profile counter updates were made atomic; no-op if atomic updates are not requested (-fprofile-update=atomic)">;
defm : ValueField<"vp-counters-per-site", "double", "1.0",
  "The average number of profile counters allocated per value profiling site">;
defm : BoolField<"vp-static-alloc", "true",
  "Do static counter allocation for value profiler">;

// LowerAllowCheckPass.cpp
defm : OptionalValueField<"lower-allow-check-percentile-cutoff-hot", "int",
  "Hot percentile cutoff">;
defm : OptionalValueField<"lower-allow-check-random-rate", "float",
  "Probability value in the range [0.0, 1.0] of unconditional pseudo-random checks">;

// MemorySanitizer.cpp
// These options allow to specify custom memory map parameters. See
// MemoryMapParams for details.
defm : ValueField<"msan-and-mask", "uint64_t", "0",
  "Define custom MSan AndMask">;
// This flag controls whether we check the shadow of the address operand of load
// or store. Such bugs are very rare, since load from a garbage address
// typically results in SEGV, but still happen (e.g. only lower bits of address
// are garbage, or the access happens early at program startup where malloc-ed
// memory is more likely to be zeroed. As of 2012-08-28 this flag adds 20%
// slowdown.
defm : BoolField<"msan-check-access-address", "true",
  "Report accesses through a pointer which has poisoned shadow">;
defm : BoolField<"msan-check-constant-shadow", "true",
  "Insert checks for constant shadow values">;
defm : BoolField<"msan-disable-checks", "false",
  "Apply no_sanitize to the whole file">;
defm : ValueField<"msan-disambiguate-warning-threshold", "int", "3",
  "Define threshold for number of checks per debug location to force origin update">;
// Currently, all the heuristically handled instructions are specifically
// IntrinsicInst. However, we use the broader "HeuristicInstructions" name to
// parallel 'msan-dump-strict-instructions', and to keep the door open to
// handling non-intrinsic instructions heuristically.
defm : BoolField<"msan-dump-heuristic-instructions", "false",
  "Prints 'unknown' instructions that were handled heuristically. Use -msan-dump-strict-instructions to print instructions that could not be handled explicitly nor heuristically">;
defm : BoolField<"msan-dump-strict-instructions", "false",
  "Print out instructions with default strict semantics i.e.,check that all the inputs are fully initialized, and mark the output as fully initialized. These semantics are applied to instructions that could not be handled explicitly nor heuristically">;
defm : OptionalBoolField<"msan-eager-checks",
  "Check arguments and return values at function call boundaries">;
// When compiling the Linux kernel, we sometimes see false positives related to
// MSan being unable to understand that inline assembly calls may initialize
// local variables. This flag makes the compiler conservatively unpoison every
// memory location passed into an assembly call. Note that this may cause false
// positives. Because it's impossible to figure out the array sizes, we can only
// unpoison the first sizeof(type) bytes for each type* pointer.
defm : BoolField<"msan-handle-asm-conservative", "true",
  "Conservative handling of inline assembly">;
defm : BoolField<"msan-handle-icmp", "true",
  "Propagate shadow through ICmpEQ and ICmpNE">;
defm : BoolField<"msan-handle-icmp-exact", "true",
  "Exact handling of relational integer ICmp">;
defm : BoolField<"msan-handle-lifetime-intrinsics", "true",
  "When possible, poison scoped variables at the beginning of the scope (slower, but more precise)">;
defm : ValueField<"msan-instrumentation-with-call-threshold", "int", "3500",
  "If the function being instrumented requires more than this number of checks and origin stores, use callbacks instead of inline checks (-1 means never use callbacks)">;
defm : OptionalBoolField<"msan-keep-going",
  "Keep going after reporting a UMR">;
defm : OptionalBoolField<"msan-kernel",
  "Enable KernelMemorySanitizer instrumentation">;
defm : OptionalValueField<"msan-origin-base", "uint64_t",
  "Define custom MSan OriginBase">;
defm : BoolField<"msan-poison-stack", "true",
  "Poison uninitialized stack variables">;
defm : ValueField<"msan-poison-stack-pattern", "int", "0xff",
  "Poison uninitialized stack variables with the given pattern">;
defm : BoolField<"msan-poison-stack-with-call", "false",
  "Poison uninitialized stack variables with a call">;
defm : BoolField<"msan-poison-undef", "true",
  "Poison fully undef temporary values. Partially undefined constant vectors are unaffected by this flag (see -msan-poison-undef-vectors)">;
defm : BoolField<"msan-poison-undef-vectors", "false",
  "Precisely poison partially undefined constant vectors. If false (legacy behavior), the entire vector is considered fully initialized, which may lead to false negatives. Fully undefined constant vectors are unaffected by this flag (see -msan-poison-undef)">;
defm : BoolField<"msan-precise-disjoint-or", "false",
  "Precisely poison disjoint OR. If false (legacy behavior), disjointedness is ignored (i.e., 1|1 is initialized)">;
defm : BoolField<"msan-print-stack-names", "true",
  "Print name of local stack variable">;
defm : OptionalValueField<"msan-shadow-base", "uint64_t",
  "Define custom MSan ShadowBase">;
defm : ValueField<"msan-switch-precision", "int", "99",
  "Controls the number of cases considered by MSan for LLVM switch instructions. 0 means no UUMs detected. Higher values lead to fewer false negatives but may impact compiler and/or application performance. N.B. LLVM switch instructions do not correspond exactly to C++ switch statements">;
// Track origins of uninitialized values.
//
// Adds a section to MemorySanitizer report that points to the allocation (stack
// or heap) the uninitialized bits came from originally.
defm : OptionalValueField<"msan-track-origins", "int",
  "Track origins (allocation sites) of poisoned memory">;
// This is off by default because of a bug in gold:
// https://sourceware.org/bugzilla/show_bug.cgi?id=19002
defm : BoolField<"msan-with-comdat", "false",
  "Place MSan constructors in comdat sections">;
defm : ValueField<"msan-xor-mask", "uint64_t", "0",
  "Define custom MSan XorMask">;

// MemProfInstrumentation.cpp
defm : ValueField<"memprof-debug", "int", "0",
  "Debug">;
defm : ValueField<"memprof-debug-func", "StringRef", "",
  "Debug func">;
defm : ValueField<"memprof-debug-max", "int", "-1",
  "Debug max inst">;
defm : ValueField<"memprof-debug-min", "int", "-1",
  "Debug min inst">;
defm : BoolField<"memprof-guard-against-version-mismatch", "true",
  "Guard against compiler/runtime version mismatch">;
defm : BoolField<"memprof-histogram", "false",
  "Collect access count histograms">;
defm : BoolField<"memprof-instrument-atomics", "true",
  "Instrument atomic instructions (rmw, cmpxchg)">;
// This flag may need to be replaced with -f[no-]memprof-reads.
defm : BoolField<"memprof-instrument-reads", "true",
  "Instrument read instructions">;
defm : BoolField<"memprof-instrument-stack", "false",
  "Instrument scalar stack variables">;
defm : BoolField<"memprof-instrument-writes", "true",
  "Instrument write instructions">;
// The shadow mapping looks like
//    Shadow = ((Mem & mask) >> scale) + offset
defm : ValueField<"memprof-mapping-granularity", "int", "64",
  "Granularity of memprof shadow mapping">;
defm : ValueField<"memprof-mapping-scale", "int", "3",
  "Scale of memprof shadow mapping">;
defm : ValueField<"memprof-memory-access-callback-prefix", "StringRef", "\"__memprof_\"",
  "Prefix for memory access callbacks">;
defm : ValueField<"memprof-runtime-default-options", "StringRef", "",
  "The default memprof options">;
defm : BoolField<"memprof-use-callbacks", "false",
  "Use callbacks instead of inline instrumentation sequences">;

// MemProfUse.cpp
defm : BoolField<"memprof-annotate-static-data-prefix", "false",
  "If true, annotate the static data section prefix">;
defm : BoolField<"memprof-attach-calleeguids", "true",
  "Attach calleeguids as value profile metadata for indirect calls">;
// By default disable matching of allocation profiles onto operator new that
// already explicitly pass a hot/cold hint, since we don't currently override
// these hints anyway.
defm : BoolField<"memprof-match-hot-cold-new", "false",
  "Match allocation profiles onto existing hot/cold operator new calls">;
defm : ValueField<"memprof-matching-cold-threshold", "unsigned", "100",
  "Min percent of cold bytes matched to hint allocation cold">;
defm : BoolField<"memprof-print-function-guids", "false",
  "Print function GUIDs computed for matching">;
defm : BoolField<"memprof-print-match-info", "false",
  "Print matching stats for each allocation context in this module's profiles">;
defm : BoolField<"memprof-print-matched-alloc-stack", "false",
  "Print full stack context for matched allocations with -memprof-print-match-info">;
defm : BoolField<"memprof-salvage-stale-profile", "false",
  "Salvage stale MemProf profile">;

// NumericalStabilitySanitizer.cpp
defm : ValueField<"check-functions-filter", "StringRef", "",
  "Only emit checks for arguments of functions whose names match the given regular expression">;
// When there is external, uninstrumented code writing to memory, the shadow
// memory can get out of sync with the application memory. Enabling this flag
// emits consistency checks for loads to catch this situation. When everything
// is instrumented, this is not strictly necessary because any load should have
// a corresponding store, but can help debug cases when the framework did a bad
// job at tracking shadow memory modifications by failing on load rather than
// store.
// TODO: provide a way to resume computations from the FT value when the load is
// inconsistent. This ensures that further computations are not polluted.
defm : BoolField<"nsan-check-loads", "false",
  "Check floating-point load">;
defm : BoolField<"nsan-check-ret", "true",
  "Check floating-point return values">;
defm : BoolField<"nsan-check-stores", "true",
  "Check floating-point stores">;
defm : BoolField<"nsan-instrument-fcmp", "true",
  "Instrument floating-point comparisons">;
// LLVM may store constant floats as bitcasted ints. It's not really necessary
// to shadow such stores, if the shadow value is unknown the framework will
// re-extend it on load anyway. Moreover, because of size collisions (e.g. bf16
// vs f16) it is impossible to determine the floating-point type based on the
// size. However, for debugging purposes it can be useful to model such stores.
defm : BoolField<"nsan-propagate-non-ft-const-stores-as-ft", "false",
  "Propagate non floating-point const stores as floating point values.For debugging purposes only">;
// Using smaller shadow types types can help improve speed. For example, `dlq`
// is 3x slower to 5x faster in opt mode and 2-6x faster in dbg mode compared to
// `dqq`.
defm : ValueField<"nsan-shadow-type-mapping", "StringRef", "\"dqq\"",
  "One shadow type id for each of `float`, `double`, `long double`. `d`,`l`,`q`,`e` mean double, x86_fp80, fp128 (quad) and ppc_fp128 (extended double) respectively. The default is to shadow `float` as `double`, and `double` and `x86_fp80` as `fp128`">;
defm : BoolField<"nsan-truncate-fcmp-eq", "true",
  "This flag controls the behaviour of fcmp equality comparisons.For equality comparisons such as `x == 0.0f`, we can perform the shadow check in the shadow (`x_shadow == 0.0) == (x == 0.0f)`) or app  domain (`(trunc(x_shadow) == 0.0f) == (x == 0.0f)`). This helps catch the case when `x_shadow` is accurate enough (and therefore close enough to zero) so that `trunc(x_shadow)` is zero even though both `x` and `x_shadow` are not">;

// PGOCtxProfLowering.cpp
defm : ListField<"profile-context-root", "StringRef",
  "A function name, assumed to be global, which will be treated as the root of an interesting graph, which will be profiled independently from other similar graphs">;

// PGOInstrumentation.cpp
defm : ListField<"ctx-prof-skip-callsite-instr", "StringRef",
  "Do not instrument callsites to functions in this list. Intended for testing">;
// Command line option to disable value profiling. The default is false: i.e.
// value profiling is enabled by default. This is for debug purpose.
defm : BoolField<"disable-vp", "false",
  "Disable Value Profiling">;
// Command line option to control appending FunctionHash to the name of a COMDAT
// function. This is to avoid the hash mismatch caused by the preinliner.
defm : BoolField<"do-comdat-renaming", "false",
  "Append function hash to the name of COMDAT function to avoid function hash mismatch due to the preinliner">;
// Command line option to set the maximum number of VP annotations to write to
// the metadata for a single indirect call callsite.
defm : ValueField<"icp-max-annotations", "unsigned", "3",
  "Max number of annotations for a single indirect call callsite">;
// Command line option to set the maximum number of value annotations to write
// to the metadata for a single memop intrinsic.
defm : ValueField<"memop-max-annotations", "unsigned", "4",
  "Max number of precise value annotations for a single memopintrinsic">;
// Command line option to enable/disable the warning about a hash mismatch in
// the profile data for Comdat functions, which often turns out to be false
// positive due to the pre-instrumentation inline.
defm : BoolField<"no-pgo-warn-mismatch-comdat-weak", "true",
  "The option is used to turn on/off warnings about hash mismatch for comdat or weak functions">;
defm : BoolField<"pgo-block-coverage", "false",
  "Use this option to enable basic block coverage instrumentation">;
defm : ValueField<"pgo-cold-instrument-entry-threshold", "uint64_t", "0",
  "For cold function instrumentation, skip instrumenting functions whose entry count is above the given value">;
defm : ValueField<"pgo-critical-edge-threshold", "unsigned", "20000",
  "Do not instrument functions with the number of critical edges  greater than this threshold">;
// Emit branch probability as optimization remarks.
defm : BoolField<"pgo-emit-branch-prob", "false",
  "When this option is on, the annotated branch probability will be emitted as optimization remarks: -{Rpass|pass-remarks}=pgo-instrumentation">;
defm : BoolField<"pgo-fix-entry-count", "true",
  "Fix function entry count in profile use">;
defm : BoolField<"pgo-function-entry-coverage", "false",
  "Use this option to enable function entry coverage instrumentation">;
defm : ValueField<"pgo-function-size-threshold", "unsigned", "0",
  "Do not instrument functions smaller than this threshold">;
// Command line option to enable/disable memop intrinsic call.size profiling.
defm : BoolField<"pgo-instr-memop", "true",
  "Use this option to turn on/off memory intrinsic size profiling">;
// Command line option to enable/disable select instruction instrumentation.
defm : BoolField<"pgo-instr-select", "true",
  "Use this option to turn on/off SELECT instruction instrumentation">;
defm : BoolField<"pgo-instrument-cold-function-only", "false",
  "Enable cold function only instrumentation">;
defm : OptionalBoolField<"pgo-instrument-entry",
  "Force to instrument function entry basicblock">;
defm : OptionalBoolField<"pgo-instrument-loop-entries",
  "Force to instrument loop entries">;
defm : BoolField<"pgo-temporal-instrumentation", "false",
  "Use this option to enable temporal instrumentation">;
// Command line option to specify the file to read profile from. This is mainly
// used for testing.
defm : ValueField<"pgo-test-profile-file", "StringRef", "",
  "Specify the path of profile data file. This is mainly for test purpose">;
defm : ValueField<"pgo-test-profile-remapping-file", "StringRef", "",
  "Specify the path of profile remapping file. This is mainly for test purpose">;
defm : ValueField<"pgo-trace-func-hash", "StringRef", "\"-\"",
  "Trace the hash of the function with this name">;
defm : BoolField<"pgo-treat-unknown-as-cold", "false",
  "For cold function instrumentation, treat count unknown(e.g. unprofiled) functions as cold">;
defm : BoolField<"pgo-verify-bfi", "false",
  "Print out mismatched BFI counts after setting profile metadata The print is enabled under -Rpass-analysis=pgo, or internal option -pass-remarks-analysis=pgo">;
defm : ValueField<"pgo-verify-bfi-cutoff", "unsigned", "5",
  "Set the threshold for pgo-verify-bfi: skip the counts whose profile count value is below">;
defm : ValueField<"pgo-verify-bfi-ratio", "unsigned", "2",
  "Set the threshold for pgo-verify-bfi:  only print out mismatched BFI if the difference percentage is greater than this value (in percentage)">;
defm : BoolField<"pgo-verify-hot-bfi", "false",
  "Print out the non-match BFI count if a hot raw profile count becomes non-hot, or a cold raw profile count becomes hot. The print is enabled under -Rpass-analysis=pgo, or internal option -pass-remarks-analysis=pgo">;
defm : BoolField<"pgo-view-block-coverage-graph", "false",
  "Create a dot file of CFGs with block coverage inference information">;
// Command line option to turn on CFG dot or text dump of raw profile counts.
defm : EnumField<"pgo-view-raw-counts", "PGOViewCountsType", "PGOVCT_None",
  "none,graph,text",
  ["PGOVCT_None", "PGOVCT_Graph", "PGOVCT_Text"],
  "A boolean option to show CFG dag or text with raw profile counts from profile data. See also option -pgo-view-counts. To limit graph display to only one function, use filtering option -view-bfi-func-name">;
// Command line option to enable/disable the warning about missing profile
// information.
defm : BoolField<"pgo-warn-missing-function", "false",
  "Use this option to turn on/off warnings about missing profile data for functions">;

// PGOMemOPSizeOpt.cpp
// Command line option to disable memory intrinsic optimization. The default is
// false. This is for debug purpose.
defm : BoolField<"disable-memop-opt", "false",
  "Disable optimize">;
defm : ValueField<"memop-value-prof-max-opt-size", "unsigned", "128",
  "Optimize the memop size <= this value">;
// The minimum call count to optimize memory intrinsic calls.
defm : ValueField<"pgo-memop-count-threshold", "unsigned", "1000",
  "The minimum count to optimize memory intrinsic calls">;
// Maximum number of versions for optimizing memory intrinsic call.
defm : ValueField<"pgo-memop-max-version", "unsigned", "3",
  "The max version for the optimized memory  intrinsic calls">;
defm : BoolField<"pgo-memop-optimize-memcmp-bcmp", "true",
  "Size-specialize memcmp and bcmp calls">;
// The percent threshold to optimize memory intrinsic calls.
defm : ValueField<"pgo-memop-percent-threshold", "unsigned", "40",
  "The percentage threshold for the memory intrinsic calls optimization">;
// Scale the counts from the annotation using the BB count value.
defm : BoolField<"pgo-memop-scale-count", "true",
  "Scale the memop size counts using the basic  block count value">;

// SanitizerBinaryMetadata.cpp
defm : BoolField<"sanitizer-metadata-atomics", "false",
  "Emit PCs for atomic operations">;
defm : BoolField<"sanitizer-metadata-covered", "false",
  "Emit PCs for covered functions">;
defm : BoolField<"sanitizer-metadata-nosanitize-attr", "true",
  "Mark some metadata features uncovered in functions with associated no_sanitize attributes">;
defm : BoolField<"sanitizer-metadata-uar", "false",
  "Emit PCs for start of functions that are subject for use-after-return checking">;
defm : BoolField<"sanitizer-metadata-weak-callbacks", "true",
  "Declare callbacks extern weak, and only call if non-null">;

// SanitizerCoverage.cpp
defm : BoolField<"sanitizer-coverage-control-flow", "false",
  "Collect control flow for each function">;
defm : BoolField<"sanitizer-coverage-drop-ctors", "false",
  "Do not emit module ctors for global counters">;
defm : BoolField<"sanitizer-coverage-gated-trace-callbacks", "false",
  "Gate the invocation of the tracing callbacks on a global variable. Currently only supported for trace-pc-guard and trace-cmp">;
defm : BoolField<"sanitizer-coverage-inline-8bit-counters", "false",
  "Increments 8-bit counter for every edge">;
defm : BoolField<"sanitizer-coverage-inline-bool-flag", "false",
  "Sets a boolean flag for every edge">;
defm : ValueField<"sanitizer-coverage-level", "int", "0",
  "Sanitizer Coverage. 0: none, 1: entry block, 2: all blocks, 3: all blocks and critical edges">;
// If true, we create a global variable that contains PCs of all instrumented
// BBs, put this global into a named section, and pass this section's bounds to
// __sanitizer_cov_pcs_init. This way the coverage instrumentation does not need
// to acquire the PCs at run-time. Works with trace-pc-guard,
// inline-8bit-counters, and inline-bool-flag.
defm : BoolField<"sanitizer-coverage-pc-table", "false",
  "Create a static PC table">;
defm : BoolField<"sanitizer-coverage-prune-blocks", "true",
  "Reduce the number of instrumented blocks">;
defm : BoolField<"sanitizer-coverage-stack-depth", "false",
  "Max stack depth tracing">;
defm : ValueField<"sanitizer-coverage-stack-depth-callback-min", "int", "0",
  "Max stack depth tracing should use callback and only when stack depth more than specified">;
defm : BoolField<"sanitizer-coverage-trace-compares", "false",
  "Tracing of CMP and similar instructions">;
defm : BoolField<"sanitizer-coverage-trace-divs", "false",
  "Tracing of DIV instructions">;
defm : BoolField<"sanitizer-coverage-trace-geps", "false",
  "Tracing of GEP instructions">;
defm : BoolField<"sanitizer-coverage-trace-loads", "false",
  "Tracing of load instructions">;
defm : BoolField<"sanitizer-coverage-trace-pc", "false",
  "Experimental pc tracing">;
defm : BoolField<"sanitizer-coverage-trace-pc-entry-exit", "false",
  "PC tracing with separate entry/exit callbacks">;
defm : BoolField<"sanitizer-coverage-trace-pc-guard", "false",
  "PC tracing with a guard">;
defm : BoolField<"sanitizer-coverage-trace-stores", "false",
  "Tracing of store instructions">;

// ThreadSanitizer.cpp
defm : BoolField<"tsan-compound-read-before-write", "false",
  "Emit special compound instrumentation for reads-before-writes">;
defm : BoolField<"tsan-distinguish-volatile", "false",
  "Emit special instrumentation for accesses to volatiles">;
defm : BoolField<"tsan-handle-cxx-exceptions", "true",
  "Handle C++ exceptions (insert cleanup blocks for unwinding)">;
defm : BoolField<"tsan-instrument-atomics", "true",
  "Instrument atomics">;
defm : BoolField<"tsan-instrument-func-entry-exit", "true",
  "Instrument function entry and exit">;
defm : BoolField<"tsan-instrument-memintrinsics", "true",
  "Instrument memintrinsics (memset/memcpy/memmove)">;
defm : BoolField<"tsan-instrument-memory-accesses", "true",
  "Instrument memory accesses">;
defm : BoolField<"tsan-instrument-read-before-write", "false",
  "Do not eliminate read instrumentation for read-before-writes">;
defm : BoolField<"tsan-omit-by-pointer-capturing", "true",
  "Omit accesses due to pointer capturing">;

// TypeSanitizer.cpp
defm : BoolField<"tysan-outline-instrumentation", "true",
  "Uses function calls for all TySan instrumentation, reducing ELF size">;
defm : BoolField<"tysan-verify-outlined-instrumentation", "false",
  "Check types twice with both inlined instrumentation and function calls. This verifies that they behave the same">;
defm : BoolField<"tysan-writes-always-set-type", "false",
  "Writes always set the type">;
