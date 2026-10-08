//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def PassesOptions : OptionsStruct;

// PassBuilderPipelines.cpp
defm : EnumField<"attributor-enable", "AttributorRunOption", "AttributorRunOption::NONE",
  "full,light,module,module-light,cgscc,cgscc-light,none",
  ["AttributorRunOption::FULL", "AttributorRunOption::LIGHT",
   "AttributorRunOption::MODULE", "AttributorRunOption::MODULE_LIGHT",
   "AttributorRunOption::CGSCC", "AttributorRunOption::CGSCC_LIGHT",
   "AttributorRunOption::NONE"],
  "Enable the attributor inter-procedural deduction pass">;
defm : BoolField<"disable-preinline", "false",
  "Disable pre-instrumentation inliner">;
defm : BoolField<"eagerly-invalidate-analyses", "true",
  "Eagerly invalidate more analyses in default pipelines">;
// This option is used in simplifying testing SampleFDO optimizations for
// profile loading.
defm : BoolField<"enable-chr", "true",
  "Enable control height reduction optimization (CHR)">;
defm : BoolField<"enable-constraint-elimination", "true",
  "Enable pass to eliminate conditions based on linear constraints">;
// TODO: There is a similar flag in WPD pass, we should consolidate them by
// parsing the option only once in PassBuilder and share it across both places.
defm : BoolField<"enable-devirtualize-speculatively", "false",
  "Enable speculative devirtualization optimization">;
defm : BoolField<"enable-dfa-jump-thread", "true",
  "Enable DFA jump threading">;
defm : BoolField<"enable-global-analyses", "true",
  "Enable inter-procedural analyses">;
defm : BoolField<"enable-gvn-hoist", "false",
  "Enable the GVN hoisting pass (default = off)">;
defm : BoolField<"enable-gvn-sink", "false",
  "Enable the GVN sinking pass (default = off)">;
defm : BoolField<"enable-instrumentor", "false",
  "Enable the Instrumentor Pass">;
defm : BoolField<"enable-jump-table-to-switch", "true",
  "Enable JumpTableToSwitch pass (default = true)">;
defm : BoolField<"enable-loop-flatten", "false",
  "Enable the LoopFlatten Pass">;
defm : BoolField<"enable-loop-versioning-licm", "false",
  "Enable the experimental Loop Versioning LICM pass">;
defm : BoolField<"enable-loopinterchange", "true",
  "Enable the LoopInterchange Pass">;
defm : BoolField<"enable-matrix", "false",
  "Enable lowering of the matrix intrinsics">;
defm : BoolField<"enable-merge-functions", "false",
  "Enable function merging as part of the optimization pipeline">;
defm : BoolField<"enable-mergeicmps", "true",
  "Enable MergeICmps pass in the optimization pipeline">;
defm : EnumField<"enable-ml-inliner", "InliningAdvisorMode", "InliningAdvisorMode::Default",
  "default,development,release",
  ["InliningAdvisorMode::Default", "InliningAdvisorMode::Development",
   "InliningAdvisorMode::Release"],
  "Enable ML policy for inliner. Currently trained for -Oz only">;
defm : BoolField<"enable-module-inliner", "false",
  "Enable module inliner">;
defm : BoolField<"enable-newgvn", "false",
  "Run the NewGVN pass">;
defm : BoolField<"enable-npm-pgo-inline-deferral", "true",
  "Enable inline deferral during PGO">;
defm : BoolField<"enable-partial-inlining", "false",
  "Run Partial inlining pass">;
defm : BoolField<"enable-post-pgo-loop-rotation", "true",
  "Run the loop rotation transformation after PGO instrumentation">;
defm : BoolField<"enable-sampled-instrumentation", "false",
  "Enable profile instrumentation sampling (default = off)">;
defm : BoolField<"enable-unroll-and-jam", "false",
  "Enable Unroll And Jam Pass">;
defm : BoolField<"extra-vectorizer-passes", "false",
  "Run cleanup optimization passes after vectorization">;
defm : BoolField<"flattened-profile-used", "false",
  "Indicate the sample profile being used is flattened, i.e., no inline hierarchy exists in the profile">;
defm : BoolField<"hot-cold-split", "false",
  "Enable hot-cold splitting pass">;
defm : ValueField<"instrument-cold-function-only-path", "StringRef", "",
  "File path for cold function only instrumentation(requires use with --pgo-instrument-cold-function-only)">;
defm : BoolField<"mandatory-inlining-first", "false",
  "Perform mandatory inlinings module-wide, before performing inlining">;
defm : BoolField<"opt-pipeline-trigger-crash", "false",
  "Trigger crash in optimization pipeline">;
defm : ValueField<"preinline-threshold", "int", "75",
  "Control the amount of inlining in pre-instrumentation inliner (default = 75)">;

// RunCodeGen.cpp
defm : OptionalBoolField<"force-new-pm-codegen",
  "Whether to force the NewPM on/off. Not setting the option will default to what the target prefers">;

// StandardInstrumentations.cpp
// An option that determines the colour used for elements that are only in the
// after part. Must be a colour named in appendix J of
// https://graphviz.org/pdf/dotguide.pdf
defm : ValueField<"dot-cfg-after-color", "StringRef", "\"forestgreen\"",
  "Color for dot-cfg after elements">;
// An option that determines the colour used for elements that are only in the
// before part. Must be a colour named in appendix J of
// https://graphviz.org/pdf/dotguide.pdf
defm : ValueField<"dot-cfg-before-color", "StringRef", "\"red\"",
  "Color for dot-cfg before elements">;
// An option that determines the colour used for elements that are in both the
// before and after parts. Must be a colour named in appendix J of
// https://graphviz.org/pdf/dotguide.pdf
defm : ValueField<"dot-cfg-common-color", "StringRef", "\"black\"",
  "Color for dot-cfg common elements">;
// An option that determines where the generated website file (named
// passes.html) and the associated pdf files (named diff_*.pdf) are saved.
defm : ValueField<"dot-cfg-dir", "StringRef", "\"./\"",
  "Generate dot files into specified directory for changed IRs">;
defm : BoolField<"dropped-variable-stats", "false",
  "Dump dropped debug variables stats">;
// An option for specifying an executable that will be called with the IR
// everytime it changes in the opt pipeline. It will also be called on the
// initial IR as it enters the pipeline. The executable will be passed the name
// of a temporary file containing the IR and the PassID. This may be used, for
// example, to call llc on the IR and run a test to determine which pass makes a
// change that changes the functioning of the IR. The usual modifier options
// work as expected.
defm : ValueField<"exec-on-ir-change", "StringRef", "",
  "exe called with module IR after each pass that changes it">;
defm : BoolField<"instnamer-after-each-pass", "false",
  "Name unnamed IR values before and after each new-PM pass with fresh generated names across pass callbacks">;
let MetaVarName = "<filename>" in
defm : ValueField<"ir-dump-directory", "StringRef", "",
  "If specified, IR printed using the -print-[before|after]{-all} options will be dumped into files in this directory rather than written to stderr">;
defm : ValueField<"opt-bisect-print-ir-path", "StringRef", "",
  "Print IR to path when opt-bisect-limit is reached">;
// An option that supports the -print-changed option. See the description for
// -print-changed for an explanation of the use of this option. Note that this
// option has no effect without -print-changed.
defm : BoolField<"print-before-changed", "false",
  "Print before passes that change them">;
// An option for specifying the dot used by
// print-changed=[dot-cfg | dot-cfg-quiet]
defm : ValueField<"print-changed-dot-path", "StringRef", "\"dot\"",
  "system dot used by change reporters">;
// Options to print the IR that was being processed when a pass crashes.
defm : BoolField<"print-on-crash", "false",
  "Print the last form of the IR before crash (use -print-on-crash-path to dump to a file)">;
defm : ValueField<"print-on-crash-path", "StringRef", "",
  "Print the last form of the IR before crash to a file">;
defm : BoolField<"print-pass-numbers", "false",
  "Print pass names and their ordinals">;
defm : OptionalBoolField<"verify-analysis-invalidation",
  "Verify that analyses a pass reports as preserved are unchanged (default: on with EXPENSIVE_CHECKS)">;
