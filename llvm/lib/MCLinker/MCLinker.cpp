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

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"

using namespace llvm;
#define DEBUG_TYPE "mclinker"

//==============================================================================
// MCInfo
//==============================================================================

MCInfo::MCInfo(std::unique_ptr<llvm::MachineModuleInfo> &&MachineModuleInfo,
         LLVMModuleAndContext &&ModuleAndContext,
         llvm::StringMap<const llvm::Function *> &FnNameToFnPtr,
         std::unique_ptr<llvm::TargetMachine> &&TgtMachine,
         std::unique_ptr<llvm::MCContext> &&McContext,
         std::optional<int> SplitIdx)
      : ModuleAndContext(std::move(ModuleAndContext)),
        McContext(std::move(McContext)),
        MachineModuleInfo(std::move(MachineModuleInfo)),
        FnNameToFnPtr(std::move(FnNameToFnPtr)),
        TgtMachine(std::move(TgtMachine)), SplitIdx(SplitIdx){
  std::string BufStr;
  llvm::raw_string_ostream BufOS(BufStr);
  llvm::WriteBitcodeToFile(*ModuleAndContext, BufOS);
  ModuleBuf = WritableMemoryBuffer::getNewUninitMemBuffer(BufStr.size());
  memcpy(ModuleBuf->getBufferStart(), BufStr.c_str(), BufStr.size());
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

  MachineModInfoPass =
      new llvm::MachineModuleInfoWrapperPass(&LLVMTgtMachine);
}


Expected<bool> MCLinker::linkLLVMModules(StringRef moduleName) {
  Expected<bool> createModuleResult =
      LinkedModule.create([&](llvm::LLVMContext &ctx) {
        return std::make_unique<llvm::Module>(moduleName, ctx);
      });

  if (createModuleResult.isError())
    return Error("failed to create an empty LLVMModule for MCLinker");

  llvm::Linker linker(*linkedModule);

  for (auto [i, smcInfos] : llvm::enumerate(symbolAndMCInfos)) {
    for (auto &[key, value] : smcInfos->symbolLinkageTypes)
      symbolLinkageTypes.insert({key, value});

    for (auto [j, mcInfo] : llvm::enumerate(smcInfos->mcInfos)) {
      mcInfos.push_back(mcInfo.get());

      // Modules have to be in the same LLVMContext to be linked.
      llvm::Expected<std::unique_ptr<llvm::Module>> moduleOr =
          llvm::parseBitcodeFile(
              llvm::MemoryBufferRef(
                  StringRef(mcInfo->moduleBuf->getBufferStart(),
                            mcInfo->moduleBuf->getBufferSize()),
                  ""),
              linkedModule->getContext());
      if (!moduleOr)
        return Error("failed to serialize post-llc modules");

      std::unique_ptr<llvm::Module> module = std::move(moduleOr.get());
      if (linker.linkInModule(std::move(module)))
        return Error("failed to link post-llc modules");
      mcInfo->mcContext->setUseNamesOnTempLabels(true);
    }
  }

  // Restore linkage type.
  for (llvm::GlobalValue &global : linkedModule->globals()) {
    if (!global.hasWeakLinkage())
      continue;
    auto iter = symbolLinkageTypes.find(global.getName().str());
    if (iter == symbolLinkageTypes.end())
      continue;

    global.setLinkage(iter->second);
    global.setDSOLocal(true);
  }

  for (llvm::Function &fn : linkedModule->functions()) {
    if (!fn.hasWeakLinkage())
      continue;

    auto iter = symbolLinkageTypes.find(fn.getName().str());
    if (iter == symbolLinkageTypes.end())
      continue;

    fn.setLinkage(iter->second);
    fn.setDSOLocal(true);
  }

  return {};
}

void MCLinker::prepareMachineModuleInfo(
    llvm::TargetMachine &llvmTargetMachine) {
  for (auto [i, smcInfos] : llvm::enumerate(symbolAndMCInfos)) {
    for (auto [j, mcInfo] : llvm::enumerate(smcInfos->mcInfos)) {
      // Move MachineFunctions from each split's codegen result
      // into machineModInfoPass to print out together in one .o
      llvm::DenseMap<const llvm::Function *,
                     std::unique_ptr<llvm::MachineFunction>> &machineFunctions =
          getMachineFunctionsFromMachineModuleInfo(*mcInfo->machineModuleInfo);

      llvm::StringMap<const llvm::Function *> &fnNameToFnPtr =
          mcInfo->fnNameToFnPtr;

      mcInfo->machineModuleInfo->getContext().setObjectFileInfo(
          llvmTargetMachine.getObjFileLowering());

      for (auto &fn : linkedModule->functions()) {
        if (fn.isDeclaration())
          continue;
        if (machineModInfoPass->getMMI().getMachineFunction(fn))
          continue;

        auto fnPtrIter = fnNameToFnPtr.find(fn.getName().str());
        if (fnPtrIter == fnNameToFnPtr.end())
          continue;
        auto mfPtrIter = machineFunctions.find(fnPtrIter->second);
        if (mfPtrIter == machineFunctions.end())
          continue;

        llvm::Function &origFn = mfPtrIter->second->getFunction();

        machineModInfoPass->getMMI().insertFunction(
            fn, std::move(mfPtrIter->second));

        // Restore function linkage types.
        if (!origFn.hasWeakLinkage())
          continue;

        auto iter = symbolLinkageTypes.find(fn.getName().str());
        if (iter == symbolLinkageTypes.end())
          continue;

        origFn.setLinkage(iter->second);
        origFn.setDSOLocal(true);
      }

      // Restore global variable linkage types.
      for (auto &global : mcInfo->moduleAndContext->globals()) {
        if (!global.hasWeakLinkage())
          continue;
        auto iter = symbolLinkageTypes.find(global.getName().str());
        if (iter == symbolLinkageTypes.end())
          continue;

        global.setLinkage(iter->second);
        global.setDSOLocal(true);
      }

      // Release memory as soon as possible to reduce peak memory footprint.
      mcInfo->machineModuleInfo.reset();
      mcInfo->fnNameToFnPtr.clear();
      mcInfo->moduleBuf.reset();
    }
  }
}

llvm::Module *
MCLinker::getModuleToPrintOneSplit(llvm::TargetMachine &llvmTargetMachine) {
  auto &mcInfo = symbolAndMCInfos[0]->mcInfos[0];

  llvm::DenseMap<const llvm::Function *, std::unique_ptr<llvm::MachineFunction>>
      &machineFunctions =
          getMachineFunctionsFromMachineModuleInfo(*mcInfo->machineModuleInfo);

  mcInfo->machineModuleInfo->getContext().setObjectFileInfo(
      llvmTargetMachine.getObjFileLowering());

  for (auto &fn : mcInfo->moduleAndContext->functions()) {
    if (fn.isDeclaration())
      continue;

    auto mfPtrIter = machineFunctions.find(&fn);
    if (mfPtrIter == machineFunctions.end())
      continue;

    machineModInfoPass->getMMI().insertFunction(fn,
                                                std::move(mfPtrIter->second));
  }

  mcInfo->mcContext->setUseNamesOnTempLabels(true);
  // Release memory as soon as possible to reduce peak memory footprint.
  mcInfo->machineModuleInfo.reset();
  mcInfo->fnNameToFnPtr.clear();
  mcInfo->moduleBuf.reset();
  return &(*mcInfo->moduleAndContext);
}

ErrorOr<WriteableBufferRef> MCLinker::linkAndPrint(StringRef moduleName,
                                                   bool emitAssembly) {

  llvm::TargetMachine &llvmTargetMachine =
      static_cast<llvm::TargetMachine &>(targetMachine);

  llvmTargetMachine.Options.MCOptions.AsmVerbose = options.verboseOutput;
  llvmTargetMachine.Options.MCOptions.PreserveAsmComments =
      options.verboseOutput;

  bool hasOneSplit =
      symbolAndMCInfos.size() == 1 && symbolAndMCInfos[0]->mcInfos.size() == 1;

  llvm::Module *oneSplitModule = nullptr;

  if (!hasOneSplit) {
    if (isNVPTXBackend(options)) {
      // For NVPTX backend to avoid false hit
      // with its stale AnnotationCache which is populated during both
      // llvm-opt and llc pipeline passes but is only cleared at the end of
      // codegen in AsmPrint. We need to make sure that llvm-opt and llc
      // are using the sname llvm::Module to that the cache can be properly
      // cleaned. We currently achieve this by keeping only one split for NVPTX
      // compilation.
      return Error("NVPTX compilation should have multiple splits.");
    }

    // link at llvm::Module level.
    ErrorOrSuccess lmResult = linkLLVMModules(moduleName);
    if (lmResult.isError())
      return Error(lmResult.getError());

    prepareMachineModuleInfo(llvmTargetMachine);

    // Function ordering may be changed in the linkedModule due to Linker,
    // but the original order matters for NVPTX backend to generate function
    // declaration properly to avoid use before def/decl illegal instructions.
    // Sort the linkedModule's functions back to to its original order
    // (only definition matter, declaration doesn't).
    if (isNVPTXBackend(options)) {
      linkedModule->getFunctionList().sort(
          [&](const auto &lhs, const auto &rhs) {
            if (lhs.isDeclaration() && rhs.isDeclaration())
              return true;

            if (lhs.isDeclaration())
              return false;

            if (rhs.isDeclaration())
              return true;

            auto iter1 = originalFnOrdering.find(lhs.getName());
            if (iter1 == originalFnOrdering.end())
              return true;
            auto iter2 = originalFnOrdering.find(rhs.getName());
            if (iter2 == originalFnOrdering.end())
              return true;

            return iter1->second < iter2->second;
          });
    }
  } else {
    oneSplitModule = getModuleToPrintOneSplit(llvmTargetMachine);
    oneSplitModule->setModuleIdentifier(moduleName);
  }

  // Prepare AsmPrint pipeline.
  WriteableBufferRef linkedObj = WriteableBuffer::get();

  llvm::legacy::PassManager passMgr;
  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  llvm::TargetLibraryInfoImpl targetLibInfo(llvm::Triple(options.targetTriple));

  // Add AsmPrint pass and run the pass manager.
  passMgr.add(new llvm::TargetLibraryInfoWrapperPass(targetLibInfo));
  if (KGEN::addPassesToAsmPrint(options, llvmTargetMachine, passMgr, *linkedObj,
                                emitAssembly
                                    ? llvm::CodeGenFileType::AssemblyFile
                                    : llvm::CodeGenFileType::ObjectFile,
                                true, machineModInfoPass, mcInfos)) {
    // Release some of the AsyncValue memory to avoid
    // wrong version of LLVMContext destructor being called due to
    // multiple LLVM being statically linked in dylibs that have
    // access to this code path.
    for (SymbolAndMCInfo *smcInfo : symbolAndMCInfos)
      smcInfo->clear();

    return Error("failed to add to ObjectFile Print pass");
  }

  const_cast<llvm::TargetLoweringObjectFile *>(
      llvmTargetMachine.getObjFileLowering())
      ->Initialize(machineModInfoPass->getMMI().getContext(), targetMachine);

  llvm::Module &moduleToRun = hasOneSplit ? *oneSplitModule : *linkedModule;
  passMgr.run(moduleToRun);

  // Release some of the AsyncValue memory to avoid
  // wrong version of LLVMContext destructor being called due to
  // multiple LLVM being statically linked in dylibs that have
  // access to this code path.
  for (SymbolAndMCInfo *smcInfo : symbolAndMCInfos)
    smcInfo->clear();

  return linkedObj;
}
