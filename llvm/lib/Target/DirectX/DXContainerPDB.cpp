//===- DXContainerPDB.cpp - DirectX PDB writer pass -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DirectX.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/DebugInfo/CodeView/GUID.h"
#include "llvm/DebugInfo/MSF/MSFBuilder.h"
#include "llvm/DebugInfo/PDB/Native/InfoStreamBuilder.h"
#include "llvm/DebugInfo/PDB/Native/PDBFileBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCDXContainerWriter.h"
#include "llvm/Pass.h"
#include "llvm/Support/IOSandbox.h"

using namespace llvm;

namespace {

class DXContainerPDB : public ModulePass, MCDXContainerBaseWriter {
  Module *M = nullptr;
  SmallVector<MCDXContainerPart> Parts;

  void reset() {
    M = nullptr;
    Parts.clear();
  }

public:
  static char ID;
  DXContainerPDB() : ModulePass(ID) {}

  StringRef getPassName() const override { return "DirectX PDB Emitter"; }

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool shouldSkipSection(StringRef SectionName, size_t SectionSize) override;
  ArrayRef<MCDXContainerPart> collectParts() override;
};

} // namespace

bool DXContainerPDB::shouldSkipSection(StringRef SectionName,
                                       size_t SectionSize) {
  if (MCDXContainerBaseWriter::shouldSkipSection(SectionName, SectionSize))
    return true;

  // Skip sections that are irrelevant for debug info.
  static const StringSet<> DebugSections{"ILDB", "ILDN", "HASH", "PDBI",
                                         "SRCI", "STAT", "RDAT", "VERS"};
  return !DebugSections.contains(SectionName);
}

static StringRef getGlobalData(const GlobalVariable &GV) {
  if (GV.hasInitializer())
    if (const auto *Data =
            dyn_cast<ConstantDataSequential>(GV.getInitializer()))
      return Data->getRawDataValues();
  return {};
}

ArrayRef<MCDXContainerPart> DXContainerPDB::collectParts() {
  Parts.clear();
  for (const GlobalVariable &GV : M->globals()) {
    StringRef Name = GV.getSection();
    StringRef Data = getGlobalData(GV);

    if (Data.empty())
      continue;
    if (shouldSkipSection(Name, Data.size()))
      continue;

    Parts.push_back({Name, Data});
  }
  return Parts;
}

bool DXContainerPDB::runOnModule(Module &M) {
  this->M = &M;

  StringRef DebugFileName;
  ArrayRef<char> ModuleHash;
  for (const GlobalVariable &GV : M.globals()) {
    if (GV.getSection() == PdbFileNameSectionName) {
      assert(DebugFileName.empty() && "Duplicate PDBNAME section");
      DebugFileName = getGlobalData(GV);
    } else if (GV.getSection() == ModuleHashSectionName) {
      assert(ModuleHash.empty() && "Duplicate PBDHASH section");
      StringRef Data = getGlobalData(GV);
      ModuleHash = ArrayRef(Data.data(), Data.size());
    }
  }

  // PDB emission was not requested.
  if (DebugFileName.empty())
    return false;
  if (ModuleHash.empty())
    report_fatal_error("Module hash for PDB not found");

  BumpPtrAllocator Allocator;
  pdb::PDBFileBuilder Builder(Allocator);

  // DirectXShaderCompiler uses block size 512.
  if (Error Err = Builder.initialize(512))
    reportFatalInternalError(std::move(Err));

  // Reserved streams that should be empty.
  static_assert(pdb::kSpecialStreamCount == 5 &&
                "First 5 streams should be empty in DirectX PDB file");
  for (uint32_t I = 0; I < pdb::kSpecialStreamCount; ++I) {
    if (auto Err = Builder.getMsfBuilder().addStream(0).takeError())
      reportFatalInternalError(std::move(Err));
  }

  // Add DXContainer stream.
  if (auto Err = Builder.getMsfBuilder().addStream(0).takeError())
    reportFatalInternalError(std::move(Err));

  // InfoStream must be filled. Bitcode hash from HASH part is used for PDB
  // GUID.
  codeview::GUID PdbGuid;
  assert(ModuleHash.size() == std::size(PdbGuid.Guid) &&
         "Module hash length must be match GUID length");
  std::copy_n(ModuleHash.begin(), std::size(PdbGuid.Guid), PdbGuid.Guid);

  auto &InfoBuilder = Builder.getInfoBuilder();
  InfoBuilder.setAge(1);
  InfoBuilder.setGuid(PdbGuid);
  InfoBuilder.setSignature(0);
  InfoBuilder.setVersion(pdb::PdbRaw_ImplVer::PdbImplVC70);

  // Write DXContainer.
  raw_svector_ostream OS(*Builder.getDXContainerData());
  write(OS, M.getTargetTriple());

  // Write PDB file.
  // FIXME(sandboxing): Remove this by routing PDB output through the VFS.
  auto BypassSandbox = sys::sandbox::scopedDisable();
  codeview::GUID IgnoredOutGuid;
  if (Error Err = Builder.commit(DebugFileName, &IgnoredOutGuid))
    reportFatalUsageError("Couldn't write to PDB file: " +
                          Twine(toString(std::move(Err))));

  reset();

  return false;
}

char DXContainerPDB::ID = 0;
INITIALIZE_PASS(DXContainerPDB, "dxil-pdb", "DirectX PDB Emitter", false, true)

ModulePass *llvm::createDXContainerPDBPass() { return new DXContainerPDB(); }
