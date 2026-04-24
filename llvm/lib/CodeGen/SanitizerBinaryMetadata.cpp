//===- SanitizerBinaryMetadata.cpp
//----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of SanitizerBinaryMetadata.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SanitizerBinaryMetadata.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Transforms/Instrumentation/SanitizerBinaryMetadata.h"
#include <algorithm>

using namespace llvm;

namespace {
// FIXME: This pass modifies Function metadata, which is not to be done in
// MachineFunctionPass. It should probably be moved to a FunctionPass.
class MachineSanitizerBinaryMetadataLegacy : public MachineFunctionPass {
public:
  static char ID;

  MachineSanitizerBinaryMetadataLegacy();
  bool runOnMachineFunction(MachineFunction &F) override;
};

struct MachineSanitizerBinaryMetadata {
  bool run(MachineFunction &MF);
};

} // namespace

INITIALIZE_PASS(MachineSanitizerBinaryMetadataLegacy, "machine-sanmd",
                "Machine Sanitizer Binary Metadata", false, false)

char MachineSanitizerBinaryMetadataLegacy::ID = 0;
char &llvm::MachineSanitizerBinaryMetadataID =
    MachineSanitizerBinaryMetadataLegacy::ID;

MachineSanitizerBinaryMetadataLegacy::MachineSanitizerBinaryMetadataLegacy()
    : MachineFunctionPass(ID) {}

bool MachineSanitizerBinaryMetadataLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  return MachineSanitizerBinaryMetadata().run(MF);
}

PreservedAnalyses
MachineSanitizerBinaryMetadataPass::run(MachineFunction &MF,
                                        MachineFunctionAnalysisManager &MFAM) {
  if (!MachineSanitizerBinaryMetadata().run(MF))
    return PreservedAnalyses::all();

  return getMachineFunctionPassPreservedAnalyses();
}

bool MachineSanitizerBinaryMetadata::run(MachineFunction &MF) {
  MDNode *MD = MF.getFunction().getMetadata(LLVMContext::MD_pcsections);
  if (!MD)
    return false;
  const auto &Section = *cast<MDString>(MD->getOperand(0));
  if (!Section.getString().starts_with(kSanitizerBinaryMetadataCoveredSection))
    return false;
  auto &AuxMDs = *cast<MDTuple>(MD->getOperand(1));
  // Assume it currently only has features.
  assert(AuxMDs.getNumOperands() == 1);
  Constant *Features =
      cast<ConstantAsMetadata>(AuxMDs.getOperand(0))->getValue();
  if (!Features->getUniqueInteger()[kSanitizerBinaryMetadataUARBit])
    return false;
  // Calculate size of stack args for the function.
  int64_t Size = 0;
  uint64_t Align = 0;
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  for (int i = -1; i >= (int)-MFI.getNumFixedObjects(); --i) {
    Size = std::max(Size, MFI.getObjectOffset(i) + MFI.getObjectSize(i));
    Align = std::max(Align, MFI.getObjectAlign(i).value());
  }
  Size = (Size + Align - 1) & ~(Align - 1);
  if (!Size)
    return false;
  // Non-zero size, update metadata.
  auto &F = MF.getFunction();
  IRBuilder<> IRB(F.getContext());
  MDBuilder MDB(F.getContext());
  // Keep the features and append size of stack args to the metadata.
  APInt NewFeatures = Features->getUniqueInteger();
  NewFeatures.setBit(kSanitizerBinaryMetadataUARHasSizeBit);
  F.setMetadata(
      LLVMContext::MD_pcsections,
      MDB.createPCSections({{Section.getString(),
                             {IRB.getInt(NewFeatures), IRB.getInt32(Size)}}}));
  return false;
}
