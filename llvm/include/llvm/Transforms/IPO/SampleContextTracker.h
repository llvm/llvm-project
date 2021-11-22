//===- Transforms/IPO/SampleContextTracker.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file provides the interface for context-sensitive profile tracker used
/// by CSSPGO.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_SAMPLECONTEXTTRACKER_H
#define LLVM_TRANSFORMS_IPO_SAMPLECONTEXTTRACKER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Instructions.h"
#include "llvm/ProfileData/SampleProf.h"
#include <list>
#include <map>
#include <vector>

using namespace llvm;
using namespace sampleprof;

namespace llvm {

// Internal trie tree representation used for tracking context tree and sample
// profiles. The path from root node to a given node represents the context of
// that nodes' profile.
class ContextTrieNode {
public:
  ContextTrieNode(ContextTrieNode *Parent = nullptr,
                  StringRef FName = StringRef(),
                  FunctionSamples *FSamples = nullptr,
                  LineLocation CallLoc = {0, 0})
      : ParentContext(Parent), FuncName(FName), FuncSamples(FSamples),
        CallSiteLoc(CallLoc){};
  ContextTrieNode *getChildContext(const LineLocation &CallSite,
                                   StringRef ChildName);
  ContextTrieNode *getHottestChildContext(const LineLocation &CallSite);
  ContextTrieNode *getOrCreateChildContext(const LineLocation &CallSite,
                                           StringRef ChildName,
                                           bool AllowCreate = true);

  ContextTrieNode &moveToChildContext(const LineLocation &CallSite,
                                      ContextTrieNode &&NodeToMove,
                                      uint32_t ContextFramesToRemove,
                                      bool DeleteNode = true);
  void removeChildContext(const LineLocation &CallSite, StringRef ChildName);
  std::map<uint64_t, ContextTrieNode> &getAllChildContext();
  StringRef getFuncName() const;
  FunctionSamples *getFunctionSamples() const;
  void setFunctionSamples(FunctionSamples *FSamples);
  Optional<uint32_t> getFunctionSize() const;
  void addFunctionSize(uint32_t FSize);
  LineLocation getCallSiteLoc() const;
  ContextTrieNode *getParentContext() const;
  void setParentContext(ContextTrieNode *Parent);
  void dumpNode();
  void dumpTree();

private:
  static uint64_t nodeHash(StringRef ChildName, const LineLocation &Callsite);

  // Map line+discriminator location to child context
  std::map<uint64_t, ContextTrieNode> AllChildContext;

  // Link to parent context node
  ContextTrieNode *ParentContext;

  // Function name for current context
  StringRef FuncName;

  // Function Samples for current context
  FunctionSamples *FuncSamples;

  // Function size for current context
  Optional<uint32_t> FuncSize;

  // Callsite location in parent context
  LineLocation CallSiteLoc;
};

// Profile tracker that manages profiles and its associated context. It
// provides interfaces used by sample profile loader to query context profile or
// base profile for given function or location; it also manages context tree
// manipulation that is needed to accommodate inline decisions so we have
// accurate post-inline profile for functions. Internally context profiles
// are organized in a trie, with each node representing profile for specific
// calling context and the context is identified by path from root to the node.
class SampleContextTracker {
public:
  struct ProfileComparer {
    bool operator()(FunctionSamples *A, FunctionSamples *B) const {
      // Sort function profiles by the number of total samples and their
      // contexts.
      if (A->getTotalSamples() == B->getTotalSamples())
        return A->getContext() < B->getContext();
      return A->getTotalSamples() > B->getTotalSamples();
    }
  };

  // Keep profiles of a function sorted so that they will be processed/promoted
  // deterministically.
  using ContextSamplesTy = std::set<FunctionSamples *, ProfileComparer>;

  SampleContextTracker(SampleProfileMap &Profiles,
                       const DenseMap<uint64_t, StringRef> *GUIDToFuncNameMap);
  // Query context profile for a specific callee with given name at a given
  // call-site. The full context is identified by location of call instruction.
  FunctionSamples *getCalleeContextSamplesFor(const CallBase &Inst,
                                              StringRef CalleeName);
  // Get samples for indirect call targets for call site at given location.
  std::vector<const FunctionSamples *>
  getIndirectCalleeContextSamplesFor(const DILocation *DIL);
  // Query context profile for a given location. The full context
  // is identified by input DILocation.
  FunctionSamples *getContextSamplesFor(const DILocation *DIL);
  // Query context profile for a given sample contxt of a function.
  FunctionSamples *getContextSamplesFor(const SampleContext &Context);
  // Get all context profile for given function.
  ContextSamplesTy &getAllContextSamplesFor(const Function &Func);
  ContextSamplesTy &getAllContextSamplesFor(StringRef Name);
  // Query base profile for a given function. A base profile is a merged view
  // of all context profiles for contexts that are not inlined.
  FunctionSamples *getBaseSamplesFor(const Function &Func,
                                     bool MergeContext = true);
  // Query base profile for a given function by name.
  FunctionSamples *getBaseSamplesFor(StringRef Name, bool MergeContext = true);
  // Retrieve the context trie node for given profile context
  ContextTrieNode *getContextFor(const SampleContext &Context);
  // Get real function name for a given trie node.
  StringRef getFuncNameFor(ContextTrieNode *Node) const;
  // Mark a context profile as inlined when function is inlined.
  // This makes sure that inlined context profile will be excluded in
  // function's base profile.
  void markContextSamplesInlined(const FunctionSamples *InlinedSamples);
  ContextTrieNode &getRootContext();
  void promoteMergeContextSamplesTree(const Instruction &Inst,
                                      StringRef CalleeName);
  // Dump the internal context profile trie.
  void dump();

private:
  ContextTrieNode *getContextFor(const DILocation *DIL);
  ContextTrieNode *getCalleeContextFor(const DILocation *DIL,
                                       StringRef CalleeName);
  ContextTrieNode *getOrCreateContextPath(const SampleContext &Context,
                                          bool AllowCreate);
  ContextTrieNode *getTopLevelContextNode(StringRef FName);
  ContextTrieNode &addTopLevelContextNode(StringRef FName);
  ContextTrieNode &promoteMergeContextSamplesTree(ContextTrieNode &NodeToPromo);
  void mergeContextNode(ContextTrieNode &FromNode, ContextTrieNode &ToNode,
                        uint32_t ContextFramesToRemove);
  ContextTrieNode &
  promoteMergeContextSamplesTree(ContextTrieNode &FromNode,
                                 ContextTrieNode &ToNodeParent,
                                 uint32_t ContextFramesToRemove);

  // Map from function name to context profiles (excluding base profile)
  StringMap<ContextSamplesTy> FuncToCtxtProfiles;

  // Map from function guid to real function names. Only used in md5 mode.
  const DenseMap<uint64_t, StringRef> *GUIDToFuncNameMap;

  // Root node for context trie tree
  ContextTrieNode RootContext;
};

} // end namespace llvm
#endif // LLVM_TRANSFORMS_IPO_SAMPLECONTEXTTRACKER_H
