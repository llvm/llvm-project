//===-- MemoryProfileInfo.cpp - memory profile info ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains utilities to analyze memory profile information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MemoryProfileInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Format.h"

using namespace llvm;
using namespace llvm::memprof;

#define DEBUG_TYPE "memory-profile-info"

namespace llvm {

cl::opt<bool> MemProfReportHintedSizes(
    "memprof-report-hinted-sizes", cl::init(false), cl::Hidden,
    cl::desc("Report total allocation sizes of hinted allocations"));

// This is useful if we have enabled reporting of hinted sizes, and want to get
// information from the indexing step for all contexts (especially for testing),
// or have specified a value less than 100% for -memprof-cloning-cold-threshold.
LLVM_ABI cl::opt<bool> MemProfKeepAllNotColdContexts(
    "memprof-keep-all-not-cold-contexts", cl::init(false), cl::Hidden,
    cl::desc("Keep all non-cold contexts (increases cloning overheads)"));

cl::opt<unsigned> MinClonedColdBytePercent(
    "memprof-cloning-cold-threshold", cl::init(100), cl::Hidden,
    cl::desc("Min percent of cold bytes to hint alloc cold during cloning"));

// Discard non-cold contexts if they overlap with much larger cold contexts,
// specifically, if all contexts reaching a given callsite are at least this
// percent cold byte allocations. This reduces the amount of cloning required
// to expose the cold contexts when they greatly dominate non-cold contexts.
cl::opt<unsigned> MinCallsiteColdBytePercent(
    "memprof-callsite-cold-threshold", cl::init(100), cl::Hidden,
    cl::desc("Min percent of cold bytes at a callsite to discard non-cold "
             "contexts"));

// Enable saving context size information for largest cold contexts, which can
// be used to flag contexts for more aggressive cloning and reporting.
cl::opt<unsigned> MinPercentMaxColdSize(
    "memprof-min-percent-max-cold-size", cl::init(100), cl::Hidden,
    cl::desc("Min percent of max cold bytes for critical cold context"));

LLVM_ABI cl::opt<bool> MemProfUseAmbiguousAttributes(
    "memprof-ambiguous-attributes", cl::init(true), cl::Hidden,
    cl::desc("Apply ambiguous memprof attribute to ambiguous allocations"));

} // end namespace llvm

bool llvm::memprof::metadataIncludesAllContextSizeInfo() {
  return MemProfReportHintedSizes || MinClonedColdBytePercent < 100;
}

bool llvm::memprof::metadataMayIncludeContextSizeInfo() {
  return metadataIncludesAllContextSizeInfo() || MinPercentMaxColdSize < 100;
}

bool llvm::memprof::recordContextSizeInfoForAnalysis() {
  return metadataMayIncludeContextSizeInfo() ||
         MinCallsiteColdBytePercent < 100;
}

MDNode *llvm::memprof::buildCallstackMetadata(ArrayRef<uint64_t> CallStack,
                                              LLVMContext &Ctx) {
  SmallVector<Metadata *, 8> StackVals;
  StackVals.reserve(CallStack.size());
  for (auto Id : CallStack) {
    auto *StackValMD =
        ValueAsMetadata::get(ConstantInt::get(Type::getInt64Ty(Ctx), Id));
    StackVals.push_back(StackValMD);
  }
  return MDNode::get(Ctx, StackVals);
}

MDNode *llvm::memprof::getMIBStackNode(const MDNode *MIB) {
  assert(MIB->getNumOperands() >= 2);
  // The stack metadata is the first operand of each memprof MIB metadata.
  return cast<MDNode>(MIB->getOperand(0));
}

AllocationType llvm::memprof::getMIBAllocType(const MDNode *MIB) {
  assert(MIB->getNumOperands() >= 2);
  // The allocation type is currently the second operand of each memprof
  // MIB metadata. This will need to change as we add additional allocation
  // types that can be applied based on the allocation profile data.
  auto *MDS = dyn_cast<MDString>(MIB->getOperand(1));
  assert(MDS);
  if (MDS->getString() == "cold") {
    return AllocationType::Cold;
  } else if (MDS->getString() == "hot") {
    return AllocationType::Hot;
  }
  return AllocationType::NotCold;
}

std::string llvm::memprof::getAllocTypeAttributeString(AllocationType Type) {
  switch (Type) {
  case AllocationType::NotCold:
    return "notcold";
    break;
  case AllocationType::Cold:
    return "cold";
    break;
  case AllocationType::Hot:
    return "hot";
    break;
  default:
    assert(false && "Unexpected alloc type");
  }
  llvm_unreachable("invalid alloc type");
}

bool llvm::memprof::hasSingleAllocType(uint8_t AllocTypes) {
  const unsigned NumAllocTypes = llvm::popcount(AllocTypes);
  assert(NumAllocTypes != 0);
  return NumAllocTypes == 1;
}

void llvm::memprof::removeAnyExistingAmbiguousAttribute(CallBase *CB) {
  if (!CB->hasFnAttr("memprof"))
    return;
  assert(CB->getFnAttr("memprof").getValueAsString() == "ambiguous");
  CB->removeFnAttr("memprof");
}

void llvm::memprof::addAmbiguousAttribute(CallBase *CB) {
  if (!MemProfUseAmbiguousAttributes)
    return;
  // We may have an existing ambiguous attribute if we are reanalyzing
  // after inlining.
  if (CB->hasFnAttr("memprof")) {
    assert(CB->getFnAttr("memprof").getValueAsString() == "ambiguous");
  } else {
    auto A = llvm::Attribute::get(CB->getContext(), "memprof", "ambiguous");
    CB->addFnAttr(A);
  }
}

void CallStackTrie::addCallStack(
    AllocationType AllocType, ArrayRef<uint64_t> StackIds,
    std::vector<ContextTotalSize> ContextSizeInfo) {
  bool First = true;
  CallStackTrieNode *Curr = nullptr;
  for (auto StackId : StackIds) {
    //  If this is the first stack frame, add or update alloc node.
    if (First) {
      First = false;
      if (Alloc) {
        assert(AllocStackId == StackId);
        Alloc->addAllocType(AllocType);
      } else {
        AllocStackId = StackId;
        Alloc = new CallStackTrieNode(AllocType);
      }
      Curr = Alloc;
      continue;
    }
    // Update existing caller node if it exists.
    auto [Next, Inserted] = Curr->Callers.try_emplace(StackId);
    if (!Inserted) {
      Curr = Next->second;
      Curr->addAllocType(AllocType);
      continue;
    }
    // Otherwise add a new caller node.
    auto *New = new CallStackTrieNode(AllocType);
    Next->second = New;
    Curr = New;
  }
  assert(Curr);
  llvm::append_range(Curr->ContextSizeInfo, ContextSizeInfo);
}

void CallStackTrie::addCallStack(MDNode *MIB) {
  // Note that we are building this from existing MD_memprof metadata.
  BuiltFromExistingMetadata = true;
  MDNode *StackMD = getMIBStackNode(MIB);
  assert(StackMD);
  std::vector<uint64_t> CallStack;
  CallStack.reserve(StackMD->getNumOperands());
  for (const auto &MIBStackIter : StackMD->operands()) {
    auto *StackId = mdconst::dyn_extract<ConstantInt>(MIBStackIter);
    assert(StackId);
    CallStack.push_back(StackId->getZExtValue());
  }
  std::vector<ContextTotalSize> ContextSizeInfo;
  // Collect the context size information if it exists.
  if (MIB->getNumOperands() > 2) {
    for (unsigned I = 2; I < MIB->getNumOperands(); I++) {
      MDNode *ContextSizePair = dyn_cast<MDNode>(MIB->getOperand(I));
      assert(ContextSizePair->getNumOperands() == 2);
      uint64_t FullStackId =
          mdconst::dyn_extract<ConstantInt>(ContextSizePair->getOperand(0))
              ->getZExtValue();
      uint64_t TotalSize =
          mdconst::dyn_extract<ConstantInt>(ContextSizePair->getOperand(1))
              ->getZExtValue();
      ContextSizeInfo.push_back({FullStackId, TotalSize});
    }
  }
  addCallStack(getMIBAllocType(MIB), CallStack, std::move(ContextSizeInfo));
}

static MDNode *createMIBNode(LLVMContext &Ctx, ArrayRef<uint64_t> MIBCallStack,
                             AllocationType AllocType,
                             ArrayRef<ContextTotalSize> ContextSizeInfo,
                             const uint64_t MaxColdSize,
                             bool BuiltFromExistingMetadata,
                             uint64_t &TotalBytes, uint64_t &ColdBytes) {
  SmallVector<Metadata *> MIBPayload(
      {buildCallstackMetadata(MIBCallStack, Ctx)});
  MIBPayload.push_back(
      MDString::get(Ctx, getAllocTypeAttributeString(AllocType)));

  if (ContextSizeInfo.empty()) {
    // The profile matcher should have provided context size info if there was a
    // MinCallsiteColdBytePercent < 100. Here we check >=100 to gracefully
    // handle a user-provided percent larger than 100. However, we may not have
    // this information if we built the Trie from existing MD_memprof metadata.
    assert(BuiltFromExistingMetadata || MinCallsiteColdBytePercent >= 100);
    return MDNode::get(Ctx, MIBPayload);
  }

  for (const auto &[FullStackId, TotalSize] : ContextSizeInfo) {
    TotalBytes += TotalSize;
    bool LargeColdContext = false;
    if (AllocType == AllocationType::Cold) {
      ColdBytes += TotalSize;
      // If we have the max cold context size from summary information and have
      // requested identification of contexts above a percentage of the max, see
      // if this context qualifies.
      if (MaxColdSize > 0 && MinPercentMaxColdSize < 100 &&
          TotalSize * 100 >= MaxColdSize * MinPercentMaxColdSize)
        LargeColdContext = true;
    }
    // Only add the context size info as metadata if we need it in the thin
    // link (currently if reporting of hinted sizes is enabled, we have
    // specified a threshold for marking allocations cold after cloning, or we
    // have identified this as a large cold context of interest above).
    if (metadataIncludesAllContextSizeInfo() || LargeColdContext) {
      auto *FullStackIdMD = ValueAsMetadata::get(
          ConstantInt::get(Type::getInt64Ty(Ctx), FullStackId));
      auto *TotalSizeMD = ValueAsMetadata::get(
          ConstantInt::get(Type::getInt64Ty(Ctx), TotalSize));
      auto *ContextSizeMD = MDNode::get(Ctx, {FullStackIdMD, TotalSizeMD});
      MIBPayload.push_back(ContextSizeMD);
    }
  }
  assert(TotalBytes > 0);
  return MDNode::get(Ctx, MIBPayload);
}

void CallStackTrie::collectContextSizeInfo(
    CallStackTrieNode *Node, std::vector<ContextTotalSize> &ContextSizeInfo) {
  llvm::append_range(ContextSizeInfo, Node->ContextSizeInfo);
  for (auto &Caller : Node->Callers)
    collectContextSizeInfo(Caller.second, ContextSizeInfo);
}

void CallStackTrie::convertHotToNotCold(CallStackTrieNode *Node) {
  if (Node->hasAllocType(AllocationType::Hot)) {
    Node->removeAllocType(AllocationType::Hot);
    Node->addAllocType(AllocationType::NotCold);
  }
  for (auto &Caller : Node->Callers)
    convertHotToNotCold(Caller.second);
}

// Copy over some or all of NewMIBNodes to the SavedMIBNodes vector, depending
// on options that enable filtering out some NotCold contexts.
static void saveFilteredNewMIBNodes(std::vector<Metadata *> &NewMIBNodes,
                                    std::vector<Metadata *> &SavedMIBNodes,
                                    unsigned CallerContextLength,
                                    uint64_t TotalBytes, uint64_t ColdBytes,
                                    bool BuiltFromExistingMetadata) {
  const bool MostlyCold =
      // If we have built the Trie from existing MD_memprof metadata, we may or
      // may not have context size information (in which case ColdBytes and
      // TotalBytes are 0, which is not also guarded against below). Even if we
      // do have some context size information from the the metadata, we have
      // already gone through a round of discarding of small non-cold contexts
      // during matching, and it would be overly aggressive to do it again, and
      // we also want to maintain the same behavior with and without reporting
      // of hinted bytes enabled.
      !BuiltFromExistingMetadata && MinCallsiteColdBytePercent < 100 &&
      ColdBytes > 0 &&
      ColdBytes * 100 >= MinCallsiteColdBytePercent * TotalBytes;

  // In the simplest case, with pruning disabled, keep all the new MIB nodes.
  if (MemProfKeepAllNotColdContexts && !MostlyCold) {
    append_range(SavedMIBNodes, NewMIBNodes);
    return;
  }

  auto EmitMessageForRemovedContexts = [](const MDNode *MIBMD, StringRef Tag,
                                          StringRef Extra) {
    assert(MIBMD->getNumOperands() > 2);
    for (unsigned I = 2; I < MIBMD->getNumOperands(); I++) {
      MDNode *ContextSizePair = dyn_cast<MDNode>(MIBMD->getOperand(I));
      assert(ContextSizePair->getNumOperands() == 2);
      uint64_t FullStackId =
          mdconst::dyn_extract<ConstantInt>(ContextSizePair->getOperand(0))
              ->getZExtValue();
      uint64_t TS =
          mdconst::dyn_extract<ConstantInt>(ContextSizePair->getOperand(1))
              ->getZExtValue();
      errs() << "MemProf hinting: Total size for " << Tag
             << " non-cold full allocation context hash " << FullStackId
             << Extra << ": " << TS << "\n";
    }
  };

  // If the cold bytes at the current callsite exceed the given threshold, we
  // discard all non-cold contexts so do not need any of the later pruning
  // handling. We can simply copy over all the cold contexts and return early.
  if (MostlyCold) {
    auto NewColdMIBNodes =
        make_filter_range(NewMIBNodes, [&](const Metadata *M) {
          auto MIBMD = cast<MDNode>(M);
          // Only append cold contexts.
          if (getMIBAllocType(MIBMD) == AllocationType::Cold)
            return true;
          if (MemProfReportHintedSizes) {
            const float PercentCold = ColdBytes * 100.0 / TotalBytes;
            std::string PercentStr;
            llvm::raw_string_ostream OS(PercentStr);
            OS << format(" for %5.2f%% cold bytes", PercentCold);
            EmitMessageForRemovedContexts(MIBMD, "discarded", OS.str());
          }
          return false;
        });
    for (auto *M : NewColdMIBNodes)
      SavedMIBNodes.push_back(M);
    return;
  }

  // Prune unneeded NotCold contexts, taking advantage of the fact
  // that we later will only clone Cold contexts, as NotCold is the allocation
  // default. We only need to keep as metadata the NotCold contexts that
  // overlap the longest with Cold allocations, so that we know how deeply we
  // need to clone. For example, assume we add the following contexts to the
  // trie:
  //    1 3 (notcold)
  //    1 2 4 (cold)
  //    1 2 5 (notcold)
  //    1 2 6 (notcold)
  // the trie looks like:
  //         1
  //        / \
  //       2   3
  //      /|\
  //     4 5 6
  //
  // It is sufficient to prune all but one not-cold contexts (either 1,2,5 or
  // 1,2,6, we arbitrarily keep the first one we encounter which will be
  // 1,2,5).
  //
  // To do this pruning, we first check if there were any not-cold
  // contexts kept for a deeper caller, which will have a context length larger
  // than the CallerContextLength being handled here (i.e. kept by a deeper
  // recursion step). If so, none of the not-cold MIB nodes added for the
  // immediate callers need to be kept. If not, we keep the first (created
  // for the immediate caller) not-cold MIB node.
  bool LongerNotColdContextKept = false;
  for (auto *MIB : NewMIBNodes) {
    auto MIBMD = cast<MDNode>(MIB);
    if (getMIBAllocType(MIBMD) == AllocationType::Cold)
      continue;
    MDNode *StackMD = getMIBStackNode(MIBMD);
    assert(StackMD);
    if (StackMD->getNumOperands() > CallerContextLength) {
      LongerNotColdContextKept = true;
      break;
    }
  }
  // Don't need to emit any for the immediate caller if we already have
  // longer overlapping contexts;
  bool KeepFirstNewNotCold = !LongerNotColdContextKept;
  auto NewColdMIBNodes = make_filter_range(NewMIBNodes, [&](const Metadata *M) {
    auto MIBMD = cast<MDNode>(M);
    // Only keep cold contexts and first (longest non-cold context).
    if (getMIBAllocType(MIBMD) != AllocationType::Cold) {
      MDNode *StackMD = getMIBStackNode(MIBMD);
      assert(StackMD);
      // Keep any already kept for longer contexts.
      if (StackMD->getNumOperands() > CallerContextLength)
        return true;
      // Otherwise keep the first one added by the immediate caller if there
      // were no longer contexts.
      if (KeepFirstNewNotCold) {
        KeepFirstNewNotCold = false;
        return true;
      }
      if (MemProfReportHintedSizes)
        EmitMessageForRemovedContexts(MIBMD, "pruned", "");
      return false;
    }
    return true;
  });
  for (auto *M : NewColdMIBNodes)
    SavedMIBNodes.push_back(M);
}

// Recursive helper to trim contexts and create metadata nodes.
// Caller should have pushed Node's loc to MIBCallStack. Doing this in the
// caller makes it simpler to handle the many early returns in this method.
// Updates the total and cold profiled bytes in the subtrie rooted at this node.
bool CallStackTrie::buildMIBNodes(CallStackTrieNode *Node, LLVMContext &Ctx,
                                  std::vector<uint64_t> &MIBCallStack,
                                  std::vector<Metadata *> &MIBNodes,
                                  bool CalleeHasAmbiguousCallerContext,
                                  uint64_t &TotalBytes, uint64_t &ColdBytes) {
  // Trim context below the first node in a prefix with a single alloc type.
  // Add an MIB record for the current call stack prefix.
  if (hasSingleAllocType(Node->AllocTypes)) {
    std::vector<ContextTotalSize> ContextSizeInfo;
    collectContextSizeInfo(Node, ContextSizeInfo);
    MIBNodes.push_back(createMIBNode(
        Ctx, MIBCallStack, (AllocationType)Node->AllocTypes, ContextSizeInfo,
        MaxColdSize, BuiltFromExistingMetadata, TotalBytes, ColdBytes));
    return true;
  }

  // We don't have a single allocation for all the contexts sharing this prefix,
  // so recursively descend into callers in trie.
  if (!Node->Callers.empty()) {
    bool NodeHasAmbiguousCallerContext = Node->Callers.size() > 1;
    bool AddedMIBNodesForAllCallerContexts = true;
    // Accumulate all new MIB nodes by the recursive calls below into a vector
    // that will later be filtered before adding to the caller's MIBNodes
    // vector.
    std::vector<Metadata *> NewMIBNodes;
    // Determine the total and cold byte counts for all callers, then add to the
    // caller's counts further below.
    uint64_t CallerTotalBytes = 0;
    uint64_t CallerColdBytes = 0;
    for (auto &Caller : Node->Callers) {
      MIBCallStack.push_back(Caller.first);
      AddedMIBNodesForAllCallerContexts &= buildMIBNodes(
          Caller.second, Ctx, MIBCallStack, NewMIBNodes,
          NodeHasAmbiguousCallerContext, CallerTotalBytes, CallerColdBytes);
      // Remove Caller.
      MIBCallStack.pop_back();
    }
    // Pass in the stack length of the MIB nodes added for the immediate caller,
    // which is the current stack length plus 1.
    saveFilteredNewMIBNodes(NewMIBNodes, MIBNodes, MIBCallStack.size() + 1,
                            CallerTotalBytes, CallerColdBytes,
                            BuiltFromExistingMetadata);
    TotalBytes += CallerTotalBytes;
    ColdBytes += CallerColdBytes;

    if (AddedMIBNodesForAllCallerContexts)
      return true;
    // We expect that the callers should be forced to add MIBs to disambiguate
    // the context in this case (see below).
    assert(!NodeHasAmbiguousCallerContext);
  }

  // If we reached here, then this node does not have a single allocation type,
  // and we didn't add metadata for a longer call stack prefix including any of
  // Node's callers. That means we never hit a single allocation type along all
  // call stacks with this prefix. This can happen due to recursion collapsing
  // or the stack being deeper than tracked by the profiler runtime, leading to
  // contexts with different allocation types being merged. In that case, we
  // trim the context just below the deepest context split, which is this
  // node if the callee has an ambiguous caller context (multiple callers),
  // since the recursive calls above returned false. Conservatively give it
  // non-cold allocation type.
  if (!CalleeHasAmbiguousCallerContext)
    return false;
  std::vector<ContextTotalSize> ContextSizeInfo;
  collectContextSizeInfo(Node, ContextSizeInfo);
  MIBNodes.push_back(createMIBNode(
      Ctx, MIBCallStack, AllocationType::NotCold, ContextSizeInfo, MaxColdSize,
      BuiltFromExistingMetadata, TotalBytes, ColdBytes));
  return true;
}

void CallStackTrie::addSingleAllocTypeAttribute(CallBase *CI, AllocationType AT,
                                                StringRef Descriptor) {
  auto AllocTypeString = getAllocTypeAttributeString(AT);
  auto A = llvm::Attribute::get(CI->getContext(), "memprof", AllocTypeString);
  // After inlining we may be able to convert an existing ambiguous allocation
  // to an unambiguous one.
  removeAnyExistingAmbiguousAttribute(CI);
  CI->addFnAttr(A);
  if (MemProfReportHintedSizes) {
    std::vector<ContextTotalSize> ContextSizeInfo;
    collectContextSizeInfo(Alloc, ContextSizeInfo);
    for (const auto &[FullStackId, TotalSize] : ContextSizeInfo) {
      errs() << "MemProf hinting: Total size for full allocation context hash "
             << FullStackId << " and " << Descriptor << " alloc type "
             << getAllocTypeAttributeString(AT) << ": " << TotalSize << "\n";
    }
  }
  if (ORE)
    ORE->emit(OptimizationRemark(DEBUG_TYPE, "MemprofAttribute", CI)
              << ore::NV("AllocationCall", CI) << " in function "
              << ore::NV("Caller", CI->getFunction())
              << " marked with memprof allocation attribute "
              << ore::NV("Attribute", AllocTypeString));
}

// Build and attach the minimal necessary MIB metadata. If the alloc has a
// single allocation type, add a function attribute instead. Returns true if
// memprof metadata attached, false if not (attribute added).
bool CallStackTrie::buildAndAttachMIBMetadata(CallBase *CI) {
  if (hasSingleAllocType(Alloc->AllocTypes)) {
    addSingleAllocTypeAttribute(CI, (AllocationType)Alloc->AllocTypes,
                                "single");
    return false;
  }
  // If there were any hot allocation contexts, the Alloc trie node would have
  // the Hot type set. If so, because we don't currently support cloning for hot
  // contexts, they should be converted to NotCold. This happens in the cloning
  // support anyway, however, doing this now enables more aggressive context
  // trimming when building the MIB metadata (and possibly may make the
  // allocation have a single NotCold allocation type), greatly reducing
  // overheads in bitcode, cloning memory and cloning time.
  if (Alloc->hasAllocType(AllocationType::Hot)) {
    convertHotToNotCold(Alloc);
    // Check whether we now have a single alloc type.
    if (hasSingleAllocType(Alloc->AllocTypes)) {
      addSingleAllocTypeAttribute(CI, (AllocationType)Alloc->AllocTypes,
                                  "single");
      return false;
    }
  }
  auto &Ctx = CI->getContext();
  std::vector<uint64_t> MIBCallStack;
  MIBCallStack.push_back(AllocStackId);
  std::vector<Metadata *> MIBNodes;
  uint64_t TotalBytes = 0;
  uint64_t ColdBytes = 0;
  assert(!Alloc->Callers.empty() && "addCallStack has not been called yet");
  // The CalleeHasAmbiguousCallerContext flag is meant to say whether the
  // callee of the given node has more than one caller. Here the node being
  // passed in is the alloc and it has no callees. So it's false.
  if (buildMIBNodes(Alloc, Ctx, MIBCallStack, MIBNodes,
                    /*CalleeHasAmbiguousCallerContext=*/false, TotalBytes,
                    ColdBytes)) {
    assert(MIBCallStack.size() == 1 &&
           "Should only be left with Alloc's location in stack");
    CI->setMetadata(LLVMContext::MD_memprof, MDNode::get(Ctx, MIBNodes));
    addAmbiguousAttribute(CI);
    return true;
  }
  // If there exists corner case that CallStackTrie has one chain to leaf
  // and all node in the chain have multi alloc type, conservatively give
  // it non-cold allocation type.
  // FIXME: Avoid this case before memory profile created. Alternatively, select
  // hint based on fraction cold.
  addSingleAllocTypeAttribute(CI, AllocationType::NotCold, "indistinguishable");
  return false;
}

template <>
CallStack<MDNode, MDNode::op_iterator>::CallStackIterator::CallStackIterator(
    const MDNode *N, bool End)
    : N(N) {
  if (!N)
    return;
  Iter = End ? N->op_end() : N->op_begin();
}

template <>
uint64_t
CallStack<MDNode, MDNode::op_iterator>::CallStackIterator::operator*() {
  assert(Iter != N->op_end());
  ConstantInt *StackIdCInt = mdconst::dyn_extract<ConstantInt>(*Iter);
  assert(StackIdCInt);
  return StackIdCInt->getZExtValue();
}

template <> uint64_t CallStack<MDNode, MDNode::op_iterator>::back() const {
  assert(N);
  return mdconst::dyn_extract<ConstantInt>(N->operands().back())
      ->getZExtValue();
}

MDNode *MDNode::getMergedMemProfMetadata(MDNode *A, MDNode *B) {
  // TODO: Support more sophisticated merging, such as selecting the one with
  // more bytes allocated, or implement support for carrying multiple allocation
  // leaf contexts. For now, keep the first one.
  if (A)
    return A;
  return B;
}

MDNode *MDNode::getMergedCallsiteMetadata(MDNode *A, MDNode *B) {
  // TODO: Support more sophisticated merging, which will require support for
  // carrying multiple contexts. For now, keep the first one.
  if (A)
    return A;
  return B;
}
