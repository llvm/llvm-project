//===--- MCLinker.cpp - MCLinker --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/MCLinker/MCLinker.h"
#include "MCLinkerUtils.h"
#include "llvm/MCLinker/MCPipeline.h"

#include "MCLinkerUtils.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;
#define DEBUG_TYPE "mclinker"

//==============================================================================
// MCInfo
//==============================================================================

MCInfo::MCInfo(std::unique_ptr<llvm::MachineModuleInfo> &&MachineModuleInfo,
               LLVMModuleAndContext &&MAndContext,
               std::unique_ptr<llvm::TargetMachine> &&TgtMachine,
               std::unique_ptr<llvm::MCContext> &&McContext,
               std::optional<int> SplitIdx)
    : ModuleAndContext(std::move(MAndContext)),
      McContext(std::move(McContext)),
      MachineModuleInfo(std::move(MachineModuleInfo)),
      TgtMachine(std::move(TgtMachine)), SplitIdx(SplitIdx) {
  std::string BufStr;
  llvm::raw_string_ostream BufOS(BufStr);
  llvm::WriteBitcodeToFile(*ModuleAndContext, BufOS);
  ModuleBuf = WritableMemoryBuffer::getNewUninitMemBuffer(BufStr.size());
  memcpy(ModuleBuf->getBufferStart(), BufStr.c_str(), BufStr.size());
  for(Function& F: ModuleAndContext->functions()) {
    FnNameToFnPtr.insert( {F.getName(), &F});
  }
}

//==============================================================================
// SymbolAndMCInfo
//==============================================================================

void SymbolAndMCInfo::clear() {
  SymbolLinkageTypes.clear();
  McInfos.clear();
}

//==============================================================================
// MCLinker
//==============================================================================

MCLinker::MCLinker(
    SmallVectorImpl<SymbolAndMCInfo *> &SymbolAndMCInfos,
    llvm::TargetMachine &TgtMachine,
    llvm::StringMap<llvm::GlobalValue::LinkageTypes> SymbolLinkageTypes)
    : SymbolAndMCInfos(SymbolAndMCInfos), TgtMachine(TgtMachine),
      SymbolLinkageTypes(std::move(SymbolLinkageTypes)) {

  llvm::TargetMachine &LLVMTgtMachine =
      static_cast<llvm::TargetMachine &>(TgtMachine);

  MachineModInfoPass = new llvm::MachineModuleInfoWrapperPass(&LLVMTgtMachine);
}

Expected<bool> MCLinker::linkLLVMModules(StringRef moduleName) {
  Expected<bool> CreateModuleOr =
      LinkedModule.create([&](llvm::LLVMContext &ctx) {
        return std::make_unique<llvm::Module>(moduleName, ctx);
      });

  if (!CreateModuleOr) {
    return make_error<StringError>(
        "failed to create an empty LLVMModule for MCLinker",
        inconvertibleErrorCode());
  }

  llvm::Linker ModuleLinker(*LinkedModule);

  for (auto [i, SmcInfos] : llvm::enumerate(SymbolAndMCInfos)) {
    for (auto &[key, value] : SmcInfos->SymbolLinkageTypes)
      SymbolLinkageTypes.insert({key, value});

    for (auto [j, McInfo] : llvm::enumerate(SmcInfos->McInfos)) {
      McInfos.push_back(McInfo.get());

      // Modules have to be in the same LLVMContext to be linked.
      llvm::Expected<std::unique_ptr<llvm::Module>> ModuleOr =
          llvm::parseBitcodeFile(
              llvm::MemoryBufferRef(
                  StringRef(McInfo->ModuleBuf->getBufferStart(),
                            McInfo->ModuleBuf->getBufferSize()),
                  ""),
              LinkedModule->getContext());
      if (!ModuleOr) {
        return make_error<StringError>("failed to serialize post-llc modules",
                                       inconvertibleErrorCode());
      }

      std::unique_ptr<llvm::Module> M = std::move(ModuleOr.get());

      if (ModuleLinker.linkInModule(std::move(M))) {
        return make_error<StringError>("failed to link post-llc modules",
                                       inconvertibleErrorCode());
      }

      McInfo->McContext->setUseNamesOnTempLabels(true);
    }
  }

  // Restore linkage type!
  for (llvm::GlobalValue &G : LinkedModule->globals()) {
    if (!G.hasWeakLinkage())
      continue;
    auto Iter = SymbolLinkageTypes.find(G.getName().str());
    if (Iter == SymbolLinkageTypes.end())
      continue;

    G.setLinkage(Iter->second);
    G.setDSOLocal(true);
  }

  for (llvm::Function &F : LinkedModule->functions()) {
    if (!F.hasWeakLinkage())
      continue;

    auto Iter = SymbolLinkageTypes.find(F.getName().str());
    if (Iter == SymbolLinkageTypes.end())
      continue;

    F.setLinkage(Iter->second);
    F.setDSOLocal(true);
  }

  return true;
}

void MCLinker::prepareMachineModuleInfo(
    llvm::TargetMachine &llvmTargetMachine) {
  for (auto [i, SmcInfos] : llvm::enumerate(SymbolAndMCInfos)) {
    for (auto [j, McInfo] : llvm::enumerate(SmcInfos->McInfos)) {
      // Move MachineFunctions from each split's codegen result
      // into machineModInfoPass to print out together in one .o
      llvm::DenseMap<const llvm::Function *,
                     std::unique_ptr<llvm::MachineFunction>> &machineFunctions =
          llvm::mclinker::getMachineFunctionsFromMachineModuleInfo(
              *McInfo->MachineModuleInfo);

      llvm::StringMap<const llvm::Function *> &FnNameToFnPtr =
          McInfo->FnNameToFnPtr;

      McInfo->MachineModuleInfo->getContext().setObjectFileInfo(
          TgtMachine.getObjFileLowering());

      for (auto &Fn : LinkedModule->functions()) {
        if (Fn.isDeclaration())
          continue;
        if (MachineModInfoPass->getMMI().getMachineFunction(Fn))
          continue;

        auto FnPtrIter = FnNameToFnPtr.find(Fn.getName().str());
        if (FnPtrIter == FnNameToFnPtr.end())
          continue;
        auto MfPtrIter = machineFunctions.find(FnPtrIter->second);
        if (MfPtrIter == machineFunctions.end())
          continue;

        llvm::Function &OrigFn = MfPtrIter->second->getFunction();

        MachineModInfoPass->getMMI().insertFunction(
            Fn, std::move(MfPtrIter->second));

        // Restore function linkage types.
        if (!OrigFn.hasWeakLinkage())
          continue;

        auto Iter = SymbolLinkageTypes.find(Fn.getName().str());
        if (Iter == SymbolLinkageTypes.end())
          continue;

        OrigFn.setLinkage(Iter->second);
        OrigFn.setDSOLocal(true);
      }

      // Restore global variable linkage types.
      for (auto &G : McInfo->ModuleAndContext->globals()) {
        if (!G.hasWeakLinkage())
          continue;
        auto Iter = SymbolLinkageTypes.find(G.getName().str());
        if (Iter == SymbolLinkageTypes.end())
          continue;

        G.setLinkage(Iter->second);
        G.setDSOLocal(true);
      }

      // Release memory as soon as possible to reduce peak memory footprint.
      McInfo->MachineModuleInfo.reset();
      McInfo->FnNameToFnPtr.clear();
      McInfo->ModuleBuf.reset();
    }
  }
}

Expected<std::unique_ptr<WritableMemoryBuffer>>
MCLinker::linkAndPrint(StringRef ModuleName, llvm::CodeGenFileType CodegenType,
                       bool VerboseOutput) {

  llvm::TargetMachine &LLVMTgtMachine =
      static_cast<llvm::TargetMachine &>(TgtMachine);

  LLVMTgtMachine.Options.MCOptions.AsmVerbose = VerboseOutput;
  LLVMTgtMachine.Options.MCOptions.PreserveAsmComments = VerboseOutput;

  // link at llvm::Module level.
  Expected<bool> LMResultOr = linkLLVMModules(ModuleName);
  if (!LMResultOr)
    return LMResultOr.takeError();

  prepareMachineModuleInfo(LLVMTgtMachine);

  // Prepare AsmPrint pipeline.
  llvm::legacy::PassManager PassMgr;
  SmallString<1024> Buf;
  raw_svector_ostream BufOS(Buf);
  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  llvm::TargetLibraryInfoImpl TargetLibInfo(TgtMachine.getTargetTriple());

  // Add AsmPrint pass and run the pass manager.
  PassMgr.add(new llvm::TargetLibraryInfoWrapperPass(TargetLibInfo));
  if (llvm::mclinker::addPassesToAsmPrint(LLVMTgtMachine, PassMgr, BufOS,
                                          CodegenType, true, MachineModInfoPass,
                                          McInfos)) {
    // Release some of the AsyncValue memory to avoid
    // wrong version of LLVMContext destructor being called due to
    // multiple LLVM being statically linked in dylibs that have
    // access to this code path.
    for (SymbolAndMCInfo *SmcInfo : SymbolAndMCInfos)
      SmcInfo->clear();

    return make_error<StringError>("failed to add to ObjectFile Print pass",
                                   inconvertibleErrorCode());
  }

  std::unique_ptr<WritableMemoryBuffer> LinkedObj =
      WritableMemoryBuffer::getNewUninitMemBuffer(Buf.size());
  memcpy(LinkedObj->getBufferStart(), Buf.c_str(), Buf.size());

  const_cast<llvm::TargetLoweringObjectFile *>(
      LLVMTgtMachine.getObjFileLowering())
      ->Initialize(MachineModInfoPass->getMMI().getContext(), TgtMachine);

  PassMgr.run(*LinkedModule);

  // Release some of the AsyncValue memory to avoid
  // wrong version of LLVMContext destructor being called due to
  // multiple LLVM being statically linked in dylibs that have
  // access to this code path.
  for (SymbolAndMCInfo *SmcInfo : SymbolAndMCInfos)
    SmcInfo->clear();

  return LinkedObj;
}
