//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def IPOOptions : OptionsStruct;

// BlockExtractor.cpp
defm : BoolField<"extract-blocks-erase-funcs", "false",
  "Erase the existing functions">;
defm : ValueField<"extract-blocks-file", "StringRef", "",
  "A file containing list of basic blocks to extract">;

// CalledValuePropagation.cpp
// The maximum number of functions to track per lattice value. Once the number
// of functions a call site can possibly target exceeds this threshold, it's
// lattice value becomes overdefined. The number of possible lattice values is
// bounded by Ch(F, M), where F is the number of functions in the module and M
// is this option's value. As such, this value should be kept very small. We
// likely can't do anything useful for call sites with a large number of
// possible targets, anyway.
defm : ValueField<"cvp-max-functions-per-value", "unsigned", "4",
  "The maximum number of functions to track per lattice value">;

// ElimAvailExtern.cpp
// This option was originally introduced to correctly support the lowering of
// LDS variables for AMDGPU when ThinLTO is enabled. It can be utilized for
// other purposes, but make sure it is safe to do so, as privatizing global
// variables is generally not safe.
defm : OptionalValueField<"avail-extern-gv-in-addrspace-to-local", "unsigned",
  "Convert available_externally global variables into locals if they are in specificed addrspace, renaming them to avoid link-time clashes">;
defm : BoolField<"avail-extern-to-local", "false",
  "Convert available_externally into locals, renaming them to avoid link-time clashes">;

// ForceFunctionAttrs.cpp
defm : ValueField<"forceattrs-csv-path", "StringRef", "",
  "Path to CSV file containing lines of function names and attributes to add to them in the form of `f1,attr1` or `f2,attr2=str`">;

// FunctionAttrs.cpp
defm : BoolField<"enable-poison-arg-attr-prop", "true",
  "Try to propagate nonnull and nofpclass argument attributes from callsites to caller functions">;
defm : BoolField<"nofree-inference", "true",
  "Infer nofree attribute during function-attrs pass">;
defm : BoolField<"nounwind-inference", "true",
  "Infer nounwind attribute during function-attrs pass">;
// TODO: Remove the old spellings after 2027-05.
def : AliasOf<"disable-nofree-inference", "nofree-inference=false">;
def : AliasOf<"disable-nounwind-inference", "nounwind-inference=false">;

// FunctionImport.cpp
defm : BoolField<"enable-import-metadata", "false",
  "Enable import metadata like 'thinlto_src_module' and 'thinlto_src_file'">;
// Used when testing importing from distributed indexes via opt
// -function-import.
defm : BoolField<"import-all-index", "false",
  "Import all external functions in index">;
defm : BoolField<"print-imports", "false",
  "Print imported functions">;
// Summary file to use for function importing when using -function-import from
// the command line.
defm : ValueField<"summary-file", "StringRef", "",
  "The summary file to use for function importing">;

// GlobalDCE.cpp
defm : BoolField<"enable-vfe", "true",
  "Enable virtual function elimination">;

// GlobalOpt.cpp
defm : ValueField<"coldcc-rel-freq", "int", "2",
  "Maximum block frequency, expressed as a percentage of caller's entry frequency, for a call site to be considered cold for enabling coldcc">;
defm : BoolField<"enable-coldcc-stress-test", "false",
  "Enable stress test of coldcc by adding calling conv to all internal functions">;
defm : ValueField<"max-ifunc-versions", "unsigned", "5",
  "Maximum number of caller/callee versions that is allowed for using the expensive (cubic) static resolution algorithm">;
defm : BoolField<"optimize-non-fmv-callers", "true",
  "Statically resolve calls to versioned functions from non-versioned callers">;

// Instrumentor.cpp
defm : ValueField<"instrumentor-read-config-paths-file", "StringRef", "",
  "Read the instrumentor configuration file paths from the specified file (newline separated)">;
defm : ValueField<"instrumentor-write-config-file", "StringRef", "",
  "Write the instrumentor configuration into the specified JSON file">;

// LowerTypeTests.cpp
defm : BoolField<"lowertypetests-avoid-reuse", "true",
  "Try to avoid reuse of byte array addresses using aliases">;
// FIXME: Remove in clang 24.
defm : BoolField<"lowertypetests-jump-table-debug-info", "true",
  "Enable debug info generation for jump tables">;
defm : ValueField<"lowertypetests-read-summary", "StringRef", "",
  "Read summary from given textual assembly or YAML file before running pass">;
defm : EnumField<"lowertypetests-summary-action", "PassSummaryAction", "",
  "none,import,export",
  ["PassSummaryAction::None", "PassSummaryAction::Import", "PassSummaryAction::Export"],
  "What to do with the summary when running this pass">;
defm : ValueField<"lowertypetests-write-summary", "StringRef", "",
  "Write summary to given YAML file after running pass">;
// FIXME: Remove in clang 26.
defm : BoolField<"reorder-cfi-jump-tables-profiles", "true",
  "Reorder CFI jump tables using profile information">;

// MergeFunctions.cpp
// Under option -mergefunc-preserve-debug-info we:
// - Do not create a new function for a thunk.
// - Retain the debug info for a thunk's parameters (and associated
//   instructions for the debug info) from the entry block.
//   Note: -debug will display the algorithm at work.
// - Create debug-info for the call (to the shared implementation) made by
//   a thunk and its return value.
// - Erase the rest of the function, retaining the (minimally sized) entry
//   block to create a thunk.
// - Preserve a thunk's call site to point to the thunk even when both occur
//   within the same translation unit, to aid debugability. Note that this
//   behaviour differs from the underlying -mergefunc implementation which
//   modifies the thunk's call site to point to the shared implementation
//   when both occur within the same translation unit.
defm : BoolField<"mergefunc-preserve-debug-info", "false",
  "Preserve debug info in thunk when mergefunc transformations are made">;
defm : BoolField<"mergefunc-use-aliases", "false",
  "Allow mergefunc to create aliases">;
let Condition = AssertsOnly in
defm : ValueField<"mergefunc-verify", "unsigned", "0",
  "How many functions in a module could be used for MergeFunctions to pass a basic correctness check. '0' disables this check. Works only with '-debug' key">;

// ModuleInliner.cpp
defm : BoolField<"ctx-prof-promote-alwaysinline", "false",
  "If using a contextual profile in this module, and an indirect call target is marked as alwaysinline, perform indirect call promotion for that target. If multiple targets for an indirect call site fit this description, they are all promoted">;

// StripSymbols.cpp
defm : BoolField<"strip-global-constants", "false",
  "Removes debug compile units which reference to non-existing global constants">;

// WholeProgramDevirt.cpp
// TODO: This option eventually should support any public visibility vtables
// with/out LTO.
defm : BoolField<"devirtualize-speculatively", "false",
  "Enable speculative devirtualization optimization">;
defm : ValueField<"wholeprogramdevirt-branch-funnel-threshold", "unsigned", "10",
  "Maximum number of call targets per call site to enable branch funnels">;
defm : EnumField<"wholeprogramdevirt-check", "WPDCheckMode", "",
  "none,trap,fallback",
  ["WPDCheckMode::None", "WPDCheckMode::Trap", "WPDCheckMode::Fallback"],
  "Type of checking for incorrect devirtualizations">;
defm : ValueField<"wholeprogramdevirt-read-summary", "StringRef", "",
  "Read summary from given bitcode or YAML file before running pass">;
defm : EnumField<"wholeprogramdevirt-summary-action", "PassSummaryAction", "",
  "none,import,export",
  ["PassSummaryAction::None", "PassSummaryAction::Import", "PassSummaryAction::Export"],
  "What to do with the summary when running this pass">;
defm : ValueField<"wholeprogramdevirt-write-summary", "StringRef", "",
  "Write summary to given bitcode or YAML file after running pass. Output file format is deduced from extension: *.bc means writing bitcode, otherwise YAML">;
