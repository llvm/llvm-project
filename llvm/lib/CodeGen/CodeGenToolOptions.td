//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def CodeGenToolOptions : OptionsStruct;

defm : BoolField<"addrsig", "false",
  "Emit an address-significance table">;
defm : ValueField<"align-loops", "unsigned", "0",
  "Default alignment for loops">;
defm : BoolField<"basic-block-address-map", "false",
  "Emit the basic block address map section">;
defm : ValueField<"basic-block-sections", "StringRef", "\"none\"",
  "Emit basic blocks into separate sections">;
defm : BoolField<"call-graph-section", "false",
  "Emit a call graph section">;
defm : OptionalEnumField<"code-model", "CodeModel::Model",
  "tiny,small,kernel,medium,large",
  ["CodeModel::Tiny", "CodeModel::Small", "CodeModel::Kernel", "CodeModel::Medium", "CodeModel::Large"],
  "Choose code model">;
defm : OptionalBoolField<"data-sections",
  "Emit data into separate sections">;
defm : BoolField<"debug-entry-values", "false",
  "Enable debug info for the debug entry values">;
// FIXME: Doesn't have way to specify separate input and output modes.
defm : OptionalEnumField<"denormal-fp-math", "DenormalMode::DenormalModeKind",
  "ieee,preserve-sign,positive-zero,dynamic",
  ["DenormalMode::IEEE", "DenormalMode::PreserveSign", "DenormalMode::PositiveZero", "DenormalMode::Dynamic"],
  "Select which denormal numbers the code is permitted to require">;
defm : OptionalEnumField<"denormal-fp-math-f32", "DenormalMode::DenormalModeKind",
  "ieee,preserve-sign,positive-zero,dynamic",
  ["DenormalMode::IEEE", "DenormalMode::PreserveSign", "DenormalMode::PositiveZero", "DenormalMode::Dynamic"],
  "Select which denormal numbers the code is permitted to require for float">;
defm : OptionalBoolField<"disable-tail-calls",
  "Never emit tail calls">;
defm : BoolField<"emit-call-site-info", "false",
  "Emit call site debug information, if debug information is enabled">;
defm : OptionalBoolField<"emulated-tls",
  "Use emulated TLS model">;
defm : BoolField<"enable-jmc-instrument", "false",
  "Instrument functions with a call to __CheckForDebuggerJustMyCode">;
defm : OptionalBoolField<"enable-tlsdesc",
  "Enable the use of TLS Descriptors">;
defm : EnumField<"exception-model", "ExceptionHandling", "ExceptionHandling::Default",
  "default,none,dwarf,sjlj,arm,wineh,wasm,emscripten",
  ["ExceptionHandling::Default", "ExceptionHandling::None", "ExceptionHandling::DwarfCFI", "ExceptionHandling::SjLj", "ExceptionHandling::ARM", "ExceptionHandling::WinEH", "ExceptionHandling::Wasm", "ExceptionHandling::Emscripten"],
  "Exception model">;
defm : OptionalEnumField<"filetype", "CodeGenFileType",
  "asm,obj,null",
  ["CodeGenFileType::AssemblyFile", "CodeGenFileType::ObjectFile", "CodeGenFileType::Null"],
  "Choose a file type (not all types are supported by all targets)">;
defm : EnumField<"float-abi", "FloatABI::ABIType", "FloatABI::Default",
  "default,soft,hard",
  ["FloatABI::Default", "FloatABI::Soft", "FloatABI::Hard"],
  "Choose float ABI type">;
defm : BoolField<"force-dwarf-frame-section", "false",
  "Always emit a debug frame section">;
defm : OptionalEnumField<"frame-pointer", "FramePointerKind",
  "all,non-leaf,non-leaf-no-reserve,reserved,none",
  ["FramePointerKind::All", "FramePointerKind::NonLeaf", "FramePointerKind::NonLeafNoReserve", "FramePointerKind::Reserved", "FramePointerKind::None"],
  "Specify frame pointer elimination optimization">;
defm : OptionalBoolField<"function-sections",
  "Emit functions into separate sections">;
defm : BoolField<"ignore-xcoff-visibility", "false",
  "Not emit the visibility attribute for asm in AIX OS or give all symbols 'unspecified' visibility in XCOFF object file">;
defm : OptionalValueField<"large-data-threshold", "uint64_t",
  "Choose large data threshold for x86_64 medium code model">;
defm : ValueField<"march", "StringRef", "",
  "Architecture to generate code for (see --version)">;
let MetaVarName = "<a1,+a2,-a3,...>" in
defm : ListField<"mattr", "StringRef",
  "Target specific attributes (-mattr=help for details)">;
let MetaVarName = "<cpu-name>" in
defm : ValueField<"mcpu", "StringRef", "",
  "Target a specific cpu type (-mcpu=help for details)">;
let MetaVarName = "<tune-cpu-name>" in
defm : ValueField<"mtune", "StringRef", "",
  "Tune for a specific CPU microarchitecture (-mtune=help for details)">;
defm : BoolField<"mxcoff-roptr", "false",
  "When set to true, const objects with relocatable address values are put into the RO data section">;
defm : BoolField<"nozero-initialized-in-bss", "false",
  "Don't place zero-initialized symbols into bss section">;
defm : BoolField<"partition-static-data-sections", "false",
  "Partition data sections using profile information">;
defm : OptionalEnumField<"relocation-model", "Reloc::Model",
  "static,pic,dynamic-no-pic,ropi,rwpi,ropi-rwpi",
  ["Reloc::Static", "Reloc::PIC_", "Reloc::DynamicNoPIC", "Reloc::ROPI", "Reloc::RWPI", "Reloc::ROPI_RWPI"],
  "Choose relocation model">;
defm : FlagOrEnumField<"save-stats", "codegen::SaveStatsMode",
  "codegen::SaveStatsMode::None", "cwd,obj",
  ["codegen::SaveStatsMode::Cwd", "codegen::SaveStatsMode::Obj"],
  "codegen::SaveStatsMode::Cwd",
  "Save LLVM statistics to a file in the current directory (`-save-stats`/`-save-stats=cwd`) or the directory of the output file (`-save-stats=obj`)">;
defm : BoolField<"separate-named-sections", "false",
  "Use separate unique sections for named sections">;
defm : BoolField<"stack-size-section", "false",
  "Emit a section containing stack size metadata">;
defm : BoolField<"stack-symbol-ordering", "true",
  "Order local stack symbols">;
defm : BoolField<"stackrealign", "false",
  "Force align the stack to the minimum alignment">;
defm : BoolField<"strict-dwarf", "false",
  "Use strict DWARF">;
defm : EnumField<"swift-async-fp", "SwiftAsyncFramePointerMode", "SwiftAsyncFramePointerMode::Always",
  "auto,always,never",
  ["SwiftAsyncFramePointerMode::DeploymentBased", "SwiftAsyncFramePointerMode::Always", "SwiftAsyncFramePointerMode::Never"],
  "Determine when the Swift async frame pointer should be set">;
defm : BoolField<"tailcallopt", "false",
  "Turn fastcc calls into tail calls by (potentially) changing ABI">;
defm : ValueField<"tls-size", "unsigned", "0",
  "Bit size of immediate TLS offsets">;
defm : OptionalValueField<"trap-func", "StringRef",
  "Emit a call to trap function rather than a trap instruction">;
defm : BoolField<"unique-basic-block-section-names", "false",
  "Give unique names to every basic block section">;
defm : BoolField<"unique-section-names", "true",
  "Give unique names to every section">;
defm : BoolField<"use-ctors", "false",
  "Use .ctors instead of .init_array">;
defm : EnumField<"vector-library", "VectorLibrary", "VectorLibrary::NoLibrary",
  "none,Accelerate,Darwin_libsystem_m,LIBMVEC,MASSV,SVML,sleefgnuabi,ArmPL,AMDLIBM",
  ["VectorLibrary::NoLibrary", "VectorLibrary::Accelerate", "VectorLibrary::DarwinLibSystemM", "VectorLibrary::LIBMVEC", "VectorLibrary::MASSV", "VectorLibrary::SVML", "VectorLibrary::SLEEFGNUABI", "VectorLibrary::ArmPL", "VectorLibrary::AMDLIBM"],
  "Vector functions library">;
defm : BoolField<"xcoff-traceback-table", "true",
  "Emit the XCOFF traceback table">;
defm : BoolField<"xray-function-index", "true",
  "Emit xray_fn_idx section">;
