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
#include "llvm/IR/Constants.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace llvm::memprof;

#define DEBUG_TYPE "memory-profile-info"

// Upper bound on lifetime access density (accesses per byte per lifetime sec)
// for marking an allocation cold.
cl::opt<float> MemProfLifetimeAccessDensityColdThreshold(
    "memprof-lifetime-access-density-cold-threshold", cl::init(0.05),
    cl::Hidden,
    cl::desc("The threshold the lifetime access density (accesses per byte per "
             "lifetime sec) must be under to consider an allocation cold"));

// Lower bound on lifetime to mark an allocation cold (in addition to accesses
// per byte per sec above). This is to avoid pessimizing short lived objects.
cl::opt<unsigned> MemProfAveLifetimeColdThreshold(
    "memprof-ave-lifetime-cold-threshold", cl::init(200), cl::Hidden,
    cl::desc("The average lifetime (s) for an allocation to be considered "
             "cold"));

// Lower bound on average lifetime accesses density (total life time access
// density / alloc count) for marking an allocation hot.
cl::opt<unsigned> MemProfMinAveLifetimeAccessDensityHotThreshold(
    "memprof-min-ave-lifetime-access-density-hot-threshold", cl::init(1000),
    cl::Hidden,
    cl::desc("The minimum TotalLifetimeAccessDensity / AllocCount for an "
             "allocation to be considered hot"));

cl::opt<bool> MemProfReportHintedSizes(
    "memprof-report-hinted-sizes", cl::init(false), cl::Hidden,
    cl::desc("Report total allocation sizes of hinted allocations"));

AllocationType llvm::memprof::getAllocType(uint64_t TotalLifetimeAccessDensity,
                                           uint64_t AllocCount,
                                           uint64_t TotalLifetime) {
  // The access densities are multiplied by 100 to hold 2 decimal places of
  // precision, so need to divide by 100.
  if (((float)TotalLifetimeAccessDensity) / AllocCount / 100 <
          MemProfLifetimeAccessDensityColdThreshold
      // Lifetime is expected to be in ms, so convert the threshold to ms.
      && ((float)TotalLifetime) / AllocCount >=
             MemProfAveLifetimeColdThreshold * 1000)
    return AllocationType::Cold;

  // The access densities are multiplied by 100 to hold 2 decimal places of
  // precision, so need to divide by 100.
  if (((float)TotalLifetimeAccessDensity) / AllocCount / 100 >
      MemProfMinAveLifetimeAccessDensityHotThreshold)
    return AllocationType::Hot;

  return AllocationType::NotCold;
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

static void addAllocTypeAttribute(LLVMContext &Ctx, CallBase *CI,
                                  AllocationType AllocType) {
  auto AllocTypeString = getAllocTypeAttributeString(AllocType);
  auto A = llvm::Attribute::get(Ctx, "memprof", AllocTypeString);
  CI->addFnAttr(A);
}

bool llvm::memprof::hasSingleAllocType(uint8_t AllocTypes) {
  const unsigned NumAllocTypes = llvm::popcount(AllocTypes);
  assert(NumAllocTypes != 0);
  return NumAllocTypes == 1;
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
        Alloc->AllocTypes |= static_cast<uint8_t>(AllocType);
      } else {
        AllocStackId = StackId;
        Alloc = new CallStackTrieNode(AllocType);
      }
      Curr = Alloc;
      continue;
    }
    // Update existing caller node if it exists.
    auto Next = Curr->Callers.find(StackId);
    if (Next != Curr->Callers.end()) {
      Curr = Next->second;
      Curr->AllocTypes |= static_cast<uint8_t>(AllocType);
      continue;
    }
    // Otherwise add a new caller node.
    auto *New = new CallStackTrieNode(AllocType);
    Curr->Callers[StackId] = New;
    Curr = New;
  }
  assert(Curr);
  Curr->ContextSizeInfo.insert(Curr->ContextSizeInfo.end(),
                               ContextSizeInfo.begin(), ContextSizeInfo.end());
}

void CallStackTrie::addCallStack(MDNode *MIB) {
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
                             ArrayRef<ContextTotalSize> ContextSizeInfo) {
  SmallVector<Metadata *> MIBPayload(
      {buildCallstackMetadata(MIBCallStack, Ctx)});
  MIBPayload.push_back(
      MDString::get(Ctx, getAllocTypeAttributeString(AllocType)));
  if (!ContextSizeInfo.empty()) {
    for (const auto &[FullStackId, TotalSize] : ContextSizeInfo) {
      auto *FullStackIdMD = ValueAsMetadata::get(
          ConstantInt::get(Type::getInt64Ty(Ctx), FullStackId));
      auto *TotalSizeMD = ValueAsMetadata::get(
          ConstantInt::get(Type::getInt64Ty(Ctx), TotalSize));
      auto *ContextSizeMD = MDNode::get(Ctx, {FullStackIdMD, TotalSizeMD});
      MIBPayload.push_back(ContextSizeMD);
    }
  }
  return MDNode::get(Ctx, MIBPayload);
}

void CallStackTrie::collectContextSizeInfo(
    CallStackTrieNode *Node, std::vector<ContextTotalSize> &ContextSizeInfo) {
  ContextSizeInfo.insert(ContextSizeInfo.end(), Node->ContextSizeInfo.begin(),
                         Node->ContextSizeInfo.end());
  for (auto &Caller : Node->Callers)
    collectContextSizeInfo(Caller.second, ContextSizeInfo);
}

// Recursive helper to trim contexts and create metadata nodes.
// Caller should have pushed Node's loc to MIBCallStack. Doing this in the
// caller makes it simpler to handle the many early returns in this method.
bool CallStackTrie::buildMIBNodes(CallStackTrieNode *Node, LLVMContext &Ctx,
                                  std::vector<uint64_t> &MIBCallStack,
                                  std::vector<Metadata *> &MIBNodes,
                                  bool CalleeHasAmbiguousCallerContext) {
  // Trim context below the first node in a prefix with a single alloc type.
  // Add an MIB record for the current call stack prefix.
  if (hasSingleAllocType(Node->AllocTypes)) {
    std::vector<ContextTotalSize> ContextSizeInfo;
    collectContextSizeInfo(Node, ContextSizeInfo);
    MIBNodes.push_back(createMIBNode(
        Ctx, MIBCallStack, (AllocationType)Node->AllocTypes, ContextSizeInfo));
    return true;
  }

  // We don't have a single allocation for all the contexts sharing this prefix,
  // so recursively descend into callers in trie.
  if (!Node->Callers.empty()) {
    bool NodeHasAmbiguousCallerContext = Node->Callers.size() > 1;
    bool AddedMIBNodesForAllCallerContexts = true;
    for (auto &Caller : Node->Callers) {
      MIBCallStack.push_back(Caller.first);
      AddedMIBNodesForAllCallerContexts &=
          buildMIBNodes(Caller.second, Ctx, MIBCallStack, MIBNodes,
                        NodeHasAmbiguousCallerContext);
      // Remove Caller.
      MIBCallStack.pop_back();
    }
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
  MIBNodes.push_back(createMIBNode(Ctx, MIBCallStack, AllocationType::NotCold,
                                   ContextSizeInfo));
  return true;
}

void CallStackTrie::addSingleAllocTypeAttribute(CallBase *CI, AllocationType AT,
                                                StringRef Descriptor) {
  addAllocTypeAttribute(CI->getContext(), CI, AT);
  if (MemProfReportHintedSizes) {
    std::vector<ContextTotalSize> ContextSizeInfo;
    collectContextSizeInfo(Alloc, ContextSizeInfo);
    for (const auto &[FullStackId, TotalSize] : ContextSizeInfo) {
      errs() << "MemProf hinting: Total size for full allocation context hash "
             << FullStackId << " and " << Descriptor << " alloc type "
             << getAllocTypeAttributeString(AT) << ": " << TotalSize << "\n";
    }
  }
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
  auto &Ctx = CI->getContext();
  std::vector<uint64_t> MIBCallStack;
  MIBCallStack.push_back(AllocStackId);
  std::vector<Metadata *> MIBNodes;
  assert(!Alloc->Callers.empty() && "addCallStack has not been called yet");
  // The last parameter is meant to say whether the callee of the given node
  // has more than one caller. Here the node being passed in is the alloc
  // and it has no callees. So it's false.
  if (buildMIBNodes(Alloc, Ctx, MIBCallStack, MIBNodes, false)) {
    assert(MIBCallStack.size() == 1 &&
           "Should only be left with Alloc's location in stack");
    CI->setMetadata(LLVMContext::MD_memprof, MDNode::get(Ctx, MIBNodes));
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
