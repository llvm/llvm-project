//===- MCLinker.h - Linker at MC level------------- -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_MCLINKER_MCLINKER_H
#define LLVM_MCLINKER_MCLINKER_H

#include "llvm/ADT/StringMap.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/ModuleSplitter/ModuleSplitter.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
/// This file defines data structures to help linking LLVM modules
/// at MC level (right after codegen) and AsmPrint into one .o or .s file.
/// This linking is needed because we parallelize the llvm opt and
/// llc pipelines by splitting LLVMModule into multiple splits
/// with symbol linkage changes.
/// Linking at MC level helps to fix the temporary symbol linkage change,
/// deduplicate multiple symbols among the splits.
/// This allows mojo compilation to produce 1 .o file for each program
/// (instead of one .a file with multiple .o files in .a) with reduced
/// object file size (due to symbol dedup and linkage restoration).

//==============================================================================
// MCInfo
//==============================================================================

struct MCInfo {
  MCInfo(std::unique_ptr<llvm::MachineModuleInfo> &&MachineModuleInfo,
         LLVMModuleAndContext &&ModuleAndContext,
         std::unique_ptr<llvm::TargetMachine> &&TgtMachine,
         std::unique_ptr<llvm::MCContext> &&McContext,
         std::optional<int> SplitIdx);

  MCInfo(MCInfo &&Other)
      : ModuleBuf(std::move(Other.ModuleBuf)),
        ModuleAndContext(std::move(Other.ModuleAndContext)),
        McContext(std::move(Other.McContext)),
        MachineModuleInfo(std::move(Other.MachineModuleInfo)),
        FnNameToFnPtr(std::move(Other.FnNameToFnPtr)),
        TgtMachine(std::move(Other.TgtMachine)), SplitIdx(Other.SplitIdx) {}

  /// Serialize the llvm::Module into bytecode.
  //  We will deserialize it back to put into
  /// a different LLVMContext that is required for linking using llvm::Linker.
  std::unique_ptr<WritableMemoryBuffer> ModuleBuf = nullptr;

  /// Keep original module split alive because llvm::Function is kept as
  /// reference in llvm::MachineFunctions and will be used during codegen.
  LLVMModuleAndContext ModuleAndContext;

  /// ExternContext to MachineModuleInfo to work around the upstream bug
  /// with the move constructor of MachineModuleInfo.
  std::unique_ptr<llvm::MCContext> McContext;

  /// This is where all the MachineFunction live that we need for AsmPrint.
  std::unique_ptr<llvm::MachineModuleInfo> MachineModuleInfo;

  /// llvm::Function name to llvm::Function* map for concatenating the
  /// MachineFunctions map.
  llvm::StringMap<const llvm::Function *> FnNameToFnPtr;

  /// Keep targetMachine alive.
  std::unique_ptr<llvm::TargetMachine> TgtMachine;

  /// parallel llvm module split id, mostly used for debugging.
  std::optional<int> SplitIdx;
};

//==============================================================================
// SymbolAndMCInfo
//==============================================================================

struct SymbolAndMCInfo {
  SymbolAndMCInfo() = default;

  SymbolAndMCInfo(SymbolAndMCInfo &&Other)
      : SymbolLinkageTypes(std::move(Other.SymbolLinkageTypes)),
        McInfos(std::move(Other.McInfos)) {}

  /// Clear member variables explicitly.
  void clear();

  /// Book-keeping original symbol linkage type if they are changed due to
  /// splitting for parallel compilation.
  llvm::StringMap<llvm::GlobalValue::LinkageTypes> SymbolLinkageTypes;

  /// Vector of codegen results for each parallel split before AsmPrint.
  SmallVector<std::unique_ptr<MCInfo>> McInfos;
};

class MCLinker {
public:
  MCLinker(SmallVectorImpl<SymbolAndMCInfo *> &SymbolAndMCInfos,
           llvm::TargetMachine &TgtMachine,
           llvm::StringMap<llvm::GlobalValue::LinkageTypes> SymbolLinkageTypes);

  /// Link multiple MC results and AsmPrint into one .o file.
  Expected<std::unique_ptr<WritableMemoryBuffer>>
  linkAndPrint(StringRef ModuleName, llvm::CodeGenFileType CodegenType,
               bool VerboseOutput);

private:
  SmallVectorImpl<SymbolAndMCInfo *> &SymbolAndMCInfos;
  llvm::TargetMachine &TgtMachine;
  SmallVector<MCInfo *> McInfos;
  LLVMModuleAndContext LinkedModule;

  llvm::StringMap<llvm::GlobalValue::LinkageTypes> SymbolLinkageTypes;
  // llvm::StringMap<unsigned> OriginalFnOrdering;
  llvm::MachineModuleInfoWrapperPass *MachineModInfoPass = nullptr;

  /// Link llvm::Modules from each split.
  Expected<bool> linkLLVMModules(StringRef ModuleName);

  // /// Get llvm::Module and prepare MachineModuleInfoWrapperPass to print if
  // /// there is only one split.
  // llvm::Module *
  // getModuleToPrintOneSplit(llvm::TargetMachine &LlvmTgtMachine);

  /// Prepare MachineModuleInfo before AsmPrinting.
  void prepareMachineModuleInfo(llvm::TargetMachine &LlvmTgtMachine);
};

} // namespace llvm

#endif
