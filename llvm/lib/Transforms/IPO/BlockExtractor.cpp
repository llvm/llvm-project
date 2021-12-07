//===- BlockExtractor.cpp - Extracts blocks into their own functions ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass extracts the specified basic blocks from the module into their
// own functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/BlockExtractor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"

using namespace llvm;

#define DEBUG_TYPE "block-extractor"

STATISTIC(NumExtracted, "Number of basic blocks extracted");

static cl::opt<std::string> BlockExtractorFile(
    "extract-blocks-file", cl::value_desc("filename"),
    cl::desc("A file containing list of basic blocks to extract"), cl::Hidden);

static cl::opt<bool>
    BlockExtractorEraseFuncs("extract-blocks-erase-funcs",
                             cl::desc("Erase the existing functions"),
                             cl::Hidden);
namespace {
class BlockExtractor {
public:
  BlockExtractor(bool EraseFunctions) : EraseFunctions(EraseFunctions) {}
  bool runOnModule(Module &M);
  void init(const SmallVectorImpl<SmallVector<BasicBlock *, 16>>
                &GroupsOfBlocksToExtract) {
    for (const SmallVectorImpl<BasicBlock *> &GroupOfBlocks :
         GroupsOfBlocksToExtract) {
      SmallVector<BasicBlock *, 16> NewGroup;
      NewGroup.append(GroupOfBlocks.begin(), GroupOfBlocks.end());
      GroupsOfBlocks.emplace_back(NewGroup);
    }
    if (!BlockExtractorFile.empty())
      loadFile();
  }

private:
  SmallVector<SmallVector<BasicBlock *, 16>, 4> GroupsOfBlocks;
  bool EraseFunctions;
  /// Map a function name to groups of blocks.
  SmallVector<std::pair<std::string, SmallVector<std::string, 4>>, 4>
      BlocksByName;

  void loadFile();
  void splitLandingPadPreds(Function &F);
};

class BlockExtractorLegacyPass : public ModulePass {
  BlockExtractor BE;
  bool runOnModule(Module &M) override;

public:
  static char ID;
  BlockExtractorLegacyPass(const SmallVectorImpl<BasicBlock *> &BlocksToExtract,
                           bool EraseFunctions)
      : ModulePass(ID), BE(EraseFunctions) {
    // We want one group per element of the input list.
    SmallVector<SmallVector<BasicBlock *, 16>, 4> MassagedGroupsOfBlocks;
    for (BasicBlock *BB : BlocksToExtract) {
      SmallVector<BasicBlock *, 16> NewGroup;
      NewGroup.push_back(BB);
      MassagedGroupsOfBlocks.push_back(NewGroup);
    }
    BE.init(MassagedGroupsOfBlocks);
  }

  BlockExtractorLegacyPass(const SmallVectorImpl<SmallVector<BasicBlock *, 16>>
                               &GroupsOfBlocksToExtract,
                           bool EraseFunctions)
      : ModulePass(ID), BE(EraseFunctions) {
    BE.init(GroupsOfBlocksToExtract);
  }

  BlockExtractorLegacyPass()
      : BlockExtractorLegacyPass(SmallVector<BasicBlock *, 0>(), false) {}
};

} // end anonymous namespace

char BlockExtractorLegacyPass::ID = 0;
INITIALIZE_PASS(BlockExtractorLegacyPass, "extract-blocks",
                "Extract basic blocks from module", false, false)

ModulePass *llvm::createBlockExtractorPass() {
  return new BlockExtractorLegacyPass();
}
ModulePass *llvm::createBlockExtractorPass(
    const SmallVectorImpl<BasicBlock *> &BlocksToExtract, bool EraseFunctions) {
  return new BlockExtractorLegacyPass(BlocksToExtract, EraseFunctions);
}
ModulePass *llvm::createBlockExtractorPass(
    const SmallVectorImpl<SmallVector<BasicBlock *, 16>>
        &GroupsOfBlocksToExtract,
    bool EraseFunctions) {
  return new BlockExtractorLegacyPass(GroupsOfBlocksToExtract, EraseFunctions);
}

/// Gets all of the blocks specified in the input file.
void BlockExtractor::loadFile() {
  auto ErrOrBuf = MemoryBuffer::getFile(BlockExtractorFile);
  if (ErrOrBuf.getError())
    report_fatal_error("BlockExtractor couldn't load the file.");
  // Read the file.
  auto &Buf = *ErrOrBuf;
  SmallVector<StringRef, 16> Lines;
  Buf->getBuffer().split(Lines, '\n', /*MaxSplit=*/-1,
                         /*KeepEmpty=*/false);
  for (const auto &Line : Lines) {
    SmallVector<StringRef, 4> LineSplit;
    Line.split(LineSplit, ' ', /*MaxSplit=*/-1,
               /*KeepEmpty=*/false);
    if (LineSplit.empty())
      continue;
    if (LineSplit.size()!=2)
      report_fatal_error("Invalid line format, expecting lines like: 'funcname bb1[;bb2..]'");
    SmallVector<StringRef, 4> BBNames;
    LineSplit[1].split(BBNames, ';', /*MaxSplit=*/-1,
                       /*KeepEmpty=*/false);
    if (BBNames.empty())
      report_fatal_error("Missing bbs name");
    BlocksByName.push_back(
        {std::string(LineSplit[0]), {BBNames.begin(), BBNames.end()}});
  }
}

/// Extracts the landing pads to make sure all of them have only one
/// predecessor.
void BlockExtractor::splitLandingPadPreds(Function &F) {
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (!isa<InvokeInst>(&I))
        continue;
      InvokeInst *II = cast<InvokeInst>(&I);
      BasicBlock *Parent = II->getParent();
      BasicBlock *LPad = II->getUnwindDest();

      // Look through the landing pad's predecessors. If one of them ends in an
      // 'invoke', then we want to split the landing pad.
      bool Split = false;
      for (auto PredBB : predecessors(LPad)) {
        if (PredBB->isLandingPad() && PredBB != Parent &&
            isa<InvokeInst>(Parent->getTerminator())) {
          Split = true;
          break;
        }
      }

      if (!Split)
        continue;

      SmallVector<BasicBlock *, 2> NewBBs;
      SplitLandingPadPredecessors(LPad, Parent, ".1", ".2", NewBBs);
    }
  }
}

bool BlockExtractor::runOnModule(Module &M) {

  bool Changed = false;

  // Get all the functions.
  SmallVector<Function *, 4> Functions;
  for (Function &F : M) {
    splitLandingPadPreds(F);
    Functions.push_back(&F);
  }

  // Get all the blocks specified in the input file.
  unsigned NextGroupIdx = GroupsOfBlocks.size();
  GroupsOfBlocks.resize(NextGroupIdx + BlocksByName.size());
  for (const auto &BInfo : BlocksByName) {
    Function *F = M.getFunction(BInfo.first);
    if (!F)
      report_fatal_error("Invalid function name specified in the input file");
    for (const auto &BBInfo : BInfo.second) {
      auto Res = llvm::find_if(*F, [&](const BasicBlock &BB) {
        return BB.getName().equals(BBInfo);
      });
      if (Res == F->end())
        report_fatal_error("Invalid block name specified in the input file");
      GroupsOfBlocks[NextGroupIdx].push_back(&*Res);
    }
    ++NextGroupIdx;
  }

  // Extract each group of basic blocks.
  for (auto &BBs : GroupsOfBlocks) {
    SmallVector<BasicBlock *, 32> BlocksToExtractVec;
    for (BasicBlock *BB : BBs) {
      // Check if the module contains BB.
      if (BB->getParent()->getParent() != &M)
        report_fatal_error("Invalid basic block");
      LLVM_DEBUG(dbgs() << "BlockExtractor: Extracting "
                        << BB->getParent()->getName() << ":" << BB->getName()
                        << "\n");
      BlocksToExtractVec.push_back(BB);
      if (const InvokeInst *II = dyn_cast<InvokeInst>(BB->getTerminator()))
        BlocksToExtractVec.push_back(II->getUnwindDest());
      ++NumExtracted;
      Changed = true;
    }
    CodeExtractorAnalysisCache CEAC(*BBs[0]->getParent());
    Function *F = CodeExtractor(BlocksToExtractVec).extractCodeRegion(CEAC);
    if (F)
      LLVM_DEBUG(dbgs() << "Extracted group '" << (*BBs.begin())->getName()
                        << "' in: " << F->getName() << '\n');
    else
      LLVM_DEBUG(dbgs() << "Failed to extract for group '"
                        << (*BBs.begin())->getName() << "'\n");
  }

  // Erase the functions.
  if (EraseFunctions || BlockExtractorEraseFuncs) {
    for (Function *F : Functions) {
      LLVM_DEBUG(dbgs() << "BlockExtractor: Trying to delete " << F->getName()
                        << "\n");
      F->deleteBody();
    }
    // Set linkage as ExternalLinkage to avoid erasing unreachable functions.
    for (Function &F : M)
      F.setLinkage(GlobalValue::ExternalLinkage);
    Changed = true;
  }

  return Changed;
}

bool BlockExtractorLegacyPass::runOnModule(Module &M) {
  return BE.runOnModule(M);
}

PreservedAnalyses BlockExtractorPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  BlockExtractor BE(false);
  BE.init(SmallVector<SmallVector<BasicBlock *, 16>, 0>());
  return BE.runOnModule(M) ? PreservedAnalyses::none()
                           : PreservedAnalyses::all();
}
