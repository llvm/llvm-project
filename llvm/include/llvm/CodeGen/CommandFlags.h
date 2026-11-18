//===-- CommandFlags.h - Command Line Flags Interface -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains codegen-specific flags that are shared between different
// command line tools. The tools "llc" and "opt" both use this file to prevent
// flag duplication.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_COMMANDFLAGS_H
#define LLVM_CODEGEN_COMMANDFLAGS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetOptions.h"
#include <optional>
#include <string>
#include <vector>

namespace llvm {

class Module;
class AttrBuilder;
class Function;
class Triple;
class TargetMachine;

namespace cl {
class LibraryOptions;
}

namespace codegen {

LLVM_ABI std::string getMArch(const cl::LibraryOptionsContext &C);

LLVM_ABI std::string getMCPU(const cl::LibraryOptionsContext &C);

LLVM_ABI std::string getMTune(const cl::LibraryOptionsContext &C);

LLVM_ABI std::vector<std::string> getMAttrs(const cl::LibraryOptionsContext &C);

LLVM_ABI std::optional<Reloc::Model>
getExplicitRelocModel(const cl::LibraryOptionsContext &C);

LLVM_ABI std::optional<CodeModel::Model>
getExplicitCodeModel(const cl::LibraryOptionsContext &C);

LLVM_ABI std::optional<uint64_t>
getExplicitLargeDataThreshold(const cl::LibraryOptionsContext &C);

LLVM_ABI std::optional<CodeGenFileType>
getExplicitFileType(const cl::LibraryOptionsContext &C);

LLVM_ABI CodeGenFileType getFileType(const cl::LibraryOptionsContext &C);

LLVM_ABI llvm::FloatABI::ABIType
getFloatABIForCalls(const cl::LibraryOptionsContext &C);

LLVM_ABI std::optional<bool>
getExplicitDataSections(const cl::LibraryOptionsContext &C);

LLVM_ABI std::optional<bool>
getExplicitFunctionSections(const cl::LibraryOptionsContext &C);

LLVM_ABI std::optional<bool>
getExplicitEmulatedTLS(const cl::LibraryOptionsContext &C);

enum SaveStatsMode { None, Cwd, Obj };

LLVM_ABI SaveStatsMode getSaveStats(const cl::LibraryOptionsContext &C);

/// The options that a tool reading getTargetOptions passes to
/// cl::parseCommandLineOptions.
LLVM_ABI ArrayRef<const cl::LibraryOptions *> getToolOptions();

/// Tools that support subtarget tuning should create this object with static
/// storage to register the -mtune command line option.
struct RegisterMTuneFlag {
  LLVM_ABI RegisterMTuneFlag();
};

/// Tools that support stats saving should create this object with static
/// storage to register the --save-stats command line option.
struct RegisterSaveStatsFlag {
  LLVM_ABI RegisterSaveStatsFlag();
};

/// Common utility function tightly tied to the options listed here. Initializes
/// a TargetOptions object with CodeGen options and returns it.
/// \p TheTriple is used to determine the default value for options if
///    options are not explicitly specified. If those triple dependant options
///    value do not have effect for your component, a default Triple() could be
///    passed in.
LLVM_ABI TargetOptions
getTargetOptions(const cl::LibraryOptionsContext &LibraryOptions,
                 const llvm::Triple &TheTriple);

LLVM_ABI std::string getCPUStr(const cl::LibraryOptionsContext &C);

LLVM_ABI std::string getTuneCPUStr(const cl::LibraryOptionsContext &C);

LLVM_ABI std::string getFeaturesStr(const cl::LibraryOptionsContext &C);

LLVM_ABI std::vector<std::string>
getFeatureList(const cl::LibraryOptionsContext &C);

LLVM_ABI void renderBoolStringAttr(AttrBuilder &B, StringRef Name, bool Val);

/// Set function attributes of function \p F based on CPU, TuneCPU, Features,
/// and command line flags.
LLVM_ABI void setFunctionAttributes(Function &F, StringRef CPU,
                                    StringRef Features, StringRef TuneCPU = "");

/// Set function attributes of functions in Module M based on CPU,
/// TuneCPU, Features, and command line flags.
LLVM_ABI void setFunctionAttributes(Module &M, StringRef CPU,
                                    StringRef Features, StringRef TuneCPU = "");

/// Creates a TargetMachine instance with the options defined on the command
/// line. This can be used for tools that do not need further customization of
/// the TargetOptions.
LLVM_ABI Expected<std::unique_ptr<TargetMachine>> createTargetMachineForTriple(
    const cl::LibraryOptionsContext &LibraryOptions, const Triple &TargetTriple,
    CodeGenOptLevel OptLevel = CodeGenOptLevel::Default);

/// Conditionally enables the collection of LLVM statistics during the tool run,
/// based on the value of the flag. Must be called before the tool run to
/// actually collect data.
LLVM_ABI void maybeEnableStatistics(const cl::LibraryOptionsContext &C);

/// Conditionally saves the collected LLVM statistics to the received output
/// file, based on the value of the flag. Should be called after the tool run,
/// and must follow a call to `maybeEnableStatistics()` to actually have data to
/// write.
LLVM_ABI int maybeSaveStatistics(const cl::LibraryOptionsContext &C,
                                 StringRef OutputFilename, StringRef ToolName);

} // namespace codegen
} // namespace llvm

#endif // LLVM_CODEGEN_COMMANDFLAGS_H
