//===-- ProfiledCallGraph.h - Profiled Call Graph ----------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_PROFGEN_PROFILEDCALLGRAPH_H
#define LLVM_TOOLS_LLVM_PROFGEN_PROFILEDCALLGRAPH_H

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ProfileData/SampleProf.h"
#include "llvm/Transforms/IPO/SampleContextTracker.h"
#include <queue>
#include <set>
#include <string>

using namespace llvm;
using namespace sampleprof;

namespace llvm {
namespace sampleprof {

struct ProfiledCallGraphNode {
  ProfiledCallGraphNode(StringRef FName = StringRef()) : Name(FName) {}
  StringRef Name;

  struct ProfiledCallGraphNodeComparer {
    bool operator()(const ProfiledCallGraphNode *L,
                    const ProfiledCallGraphNode *R) const {
      return L->Name < R->Name;
    }
  };
  std::set<ProfiledCallGraphNode *, ProfiledCallGraphNodeComparer> Callees;
};

class ProfiledCallGraph {
public:
  using iterator = std::set<ProfiledCallGraphNode *>::iterator;
  ProfiledCallGraph(StringMap<FunctionSamples> &ProfileMap,
                    SampleContextTracker &ContextTracker) {
    // Add all profiled functions into profiled call graph.
    // We only add function with actual context profile
    for (auto &FuncSample : ProfileMap) {
      FunctionSamples *FSamples = &FuncSample.second;
      addProfiledFunction(FSamples->getName());
    }

    // BFS traverse the context profile trie to add call edges for
    // both samples calls as well as calls shown in context.
    std::queue<ContextTrieNode *> Queue;
    Queue.push(&ContextTracker.getRootContext());
    while (!Queue.empty()) {
      ContextTrieNode *Caller = Queue.front();
      Queue.pop();
      FunctionSamples *CallerSamples = Caller->getFunctionSamples();

      // Add calls for context, if both caller and callee has context profile.
      for (auto &Child : Caller->getAllChildContext()) {
        ContextTrieNode *Callee = &Child.second;
        Queue.push(Callee);
        if (CallerSamples && Callee->getFunctionSamples()) {
          addProfiledCall(Caller->getFuncName(), Callee->getFuncName());
        }
      }

      // Add calls from call site samples
      if (CallerSamples) {
        for (auto &LocCallSite : CallerSamples->getBodySamples()) {
          for (auto &NameCallSite : LocCallSite.second.getCallTargets()) {
            addProfiledCall(Caller->getFuncName(), NameCallSite.first());
          }
        }
      }
    }
  }

  iterator begin() { return Root.Callees.begin(); }
  iterator end() { return Root.Callees.end(); }
  ProfiledCallGraphNode *getEntryNode() { return &Root; }
  void addProfiledFunction(StringRef Name) {
    if (!ProfiledFunctions.count(Name)) {
      // Link to synthetic root to make sure every node is reachable
      // from root. This does not affect SCC order.
      Root.Callees.insert(&ProfiledFunctions[Name]);
      ProfiledFunctions[Name] = ProfiledCallGraphNode(Name);
    }
  }
  void addProfiledCall(StringRef CallerName, StringRef CalleeName) {
    assert(ProfiledFunctions.count(CallerName));
    auto CalleeIt = ProfiledFunctions.find(CalleeName);
    if (CalleeIt == ProfiledFunctions.end()) {
      return;
    }
    ProfiledFunctions[CallerName].Callees.insert(&CalleeIt->second);
  }

private:
  ProfiledCallGraphNode Root;
  StringMap<ProfiledCallGraphNode> ProfiledFunctions;
};

} // end namespace sampleprof

template <> struct GraphTraits<ProfiledCallGraphNode *> {
  using NodeRef = ProfiledCallGraphNode *;
  using ChildIteratorType = std::set<ProfiledCallGraphNode *>::iterator;

  static NodeRef getEntryNode(NodeRef PCGN) { return PCGN; }
  static ChildIteratorType child_begin(NodeRef N) { return N->Callees.begin(); }
  static ChildIteratorType child_end(NodeRef N) { return N->Callees.end(); }
};

template <>
struct GraphTraits<ProfiledCallGraph *>
    : public GraphTraits<ProfiledCallGraphNode *> {
  static NodeRef getEntryNode(ProfiledCallGraph *PCG) {
    return PCG->getEntryNode();
  }

  static ChildIteratorType nodes_begin(ProfiledCallGraph *PCG) {
    return PCG->begin();
  }

  static ChildIteratorType nodes_end(ProfiledCallGraph *PCG) {
    return PCG->end();
  }
};

} // end namespace llvm

#endif
