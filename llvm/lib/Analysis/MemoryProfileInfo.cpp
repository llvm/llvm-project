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
#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace llvm::memprof;

#define DEBUG_TYPE "memory-profile-info"

// Upper bound on accesses per byte for marking an allocation cold.
cl::opt<float> MemProfAccessesPerByteColdThreshold(
    "memprof-accesses-per-byte-cold-threshold", cl::init(10.0), cl::Hidden,
    cl::desc("The threshold the accesses per byte must be under to consider "
             "an allocation cold"));

// Lower bound on lifetime to mark an allocation cold (in addition to accesses
// per byte above). This is to avoid pessimizing short lived objects.
cl::opt<unsigned> MemProfMinLifetimeColdThreshold(
    "memprof-min-lifetime-cold-threshold", cl::init(200), cl::Hidden,
    cl::desc("The minimum lifetime (s) for an allocation to be considered "
             "cold"));

AllocationType llvm::memprof::getAllocType(uint64_t MaxAccessCount,
                                           uint64_t MinSize,
                                           uint64_t MinLifetime) {
  if (((float)MaxAccessCount) / MinSize < MemProfAccessesPerByteColdThreshold &&
      // MinLifetime is expected to be in ms, so convert the threshold to ms.
      MinLifetime >= MemProfMinLifetimeColdThreshold * 1000)
    return AllocationType::Cold;
  return AllocationType::NotCold;
}

MDNode *llvm::memprof::buildCallstackMetadata(ArrayRef<uint64_t> CallStack,
                                              LLVMContext &Ctx) {
  std::vector<Metadata *> StackVals;
  for (auto Id : CallStack) {
    auto *StackValMD =
        ValueAsMetadata::get(ConstantInt::get(Type::getInt64Ty(Ctx), Id));
    StackVals.push_back(StackValMD);
  }
  return MDNode::get(Ctx, StackVals);
}

MDNode *llvm::memprof::getMIBStackNode(const MDNode *MIB) {
  assert(MIB->getNumOperands() == 2);
  // The stack metadata is the first operand of each memprof MIB metadata.
  return cast<MDNode>(MIB->getOperand(0));
}

AllocationType llvm::memprof::getMIBAllocType(const MDNode *MIB) {
  assert(MIB->getNumOperands() == 2);
  // The allocation type is currently the second operand of each memprof
  // MIB metadata. This will need to change as we add additional allocation
  // types that can be applied based on the allocation profile data.
  auto *MDS = dyn_cast<MDString>(MIB->getOperand(1));
  assert(MDS);
  if (MDS->getString().equals("cold"))
    return AllocationType::Cold;
  return AllocationType::NotCold;
}

static std::string getAllocTypeAttributeString(AllocationType Type) {
  switch (Type) {
  case AllocationType::NotCold:
    return "notcold";
    break;
  case AllocationType::Cold:
    return "cold";
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

static bool hasSingleAllocType(uint8_t AllocTypes) {
  const unsigned NumAllocTypes = countPopulation(AllocTypes);
  assert(NumAllocTypes != 0);
  return NumAllocTypes == 1;
}

void CallStackTrie::addCallStack(AllocationType AllocType,
                                 ArrayRef<uint64_t> StackIds) {
  bool First = true;
  CallStackTrieNode *Curr = nullptr;
  for (auto StackId : StackIds) {
    // If this is the first stack frame, add or update alloc node.
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
}

void CallStackTrie::addCallStack(MDNode *MIB) {
  MDNode *StackMD = getMIBStackNode(MIB);
  assert(StackMD);
  std::vector<uint64_t> CallStack;
  CallStack.reserve(StackMD->getNumOperands());
  for (auto &MIBStackIter : StackMD->operands()) {
    auto *StackId = mdconst::dyn_extract<ConstantInt>(MIBStackIter);
    assert(StackId);
    CallStack.push_back(StackId->getZExtValue());
  }
  addCallStack(getMIBAllocType(MIB), CallStack);
}

static MDNode *createMIBNode(LLVMContext &Ctx,
                             std::vector<uint64_t> &MIBCallStack,
                             AllocationType AllocType) {
  std::vector<Metadata *> MIBPayload(
      {buildCallstackMetadata(MIBCallStack, Ctx)});
  MIBPayload.push_back(
      MDString::get(Ctx, getAllocTypeAttributeString(AllocType)));
  return MDNode::get(Ctx, MIBPayload);
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
    MIBNodes.push_back(
        createMIBNode(Ctx, MIBCallStack, (AllocationType)Node->AllocTypes));
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
  MIBNodes.push_back(createMIBNode(Ctx, MIBCallStack, AllocationType::NotCold));
  return true;
}

// Build and attach the minimal necessary MIB metadata. If the alloc has a
// single allocation type, add a function attribute instead. Returns true if
// memprof metadata attached, false if not (attribute added).
bool CallStackTrie::buildAndAttachMIBMetadata(CallBase *CI) {
  auto &Ctx = CI->getContext();
  if (hasSingleAllocType(Alloc->AllocTypes)) {
    addAllocTypeAttribute(Ctx, CI, (AllocationType)Alloc->AllocTypes);
    return false;
  }
  std::vector<uint64_t> MIBCallStack;
  MIBCallStack.push_back(AllocStackId);
  std::vector<Metadata *> MIBNodes;
  assert(!Alloc->Callers.empty() && "addCallStack has not been called yet");
  buildMIBNodes(Alloc, Ctx, MIBCallStack, MIBNodes,
                /*CalleeHasAmbiguousCallerContext=*/true);
  assert(MIBCallStack.size() == 1 &&
         "Should only be left with Alloc's location in stack");
  CI->setMetadata(LLVMContext::MD_memprof, MDNode::get(Ctx, MIBNodes));
  return true;
}
