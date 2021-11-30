//===-- ProfiledCallGraph.h - Profiled Call Graph ----------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_PROFILEDCALLGRAPH_H
#define LLVM_TRANSFORMS_IPO_PROFILEDCALLGRAPH_H

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ProfileData/SampleProf.h"
#include "llvm/ProfileData/SampleProfReader.h"
#include "llvm/Transforms/IPO/SampleContextTracker.h"
#include <queue>
#include <set>

using namespace llvm;
using namespace sampleprof;

namespace llvm {
namespace sampleprof {

struct ProfiledCallGraphNode;

struct ProfiledCallGraphEdge {
  ProfiledCallGraphEdge(ProfiledCallGraphNode *Source,
                        ProfiledCallGraphNode *Target, uint64_t Weight)
      : Source(Source), Target(Target), Weight(Weight) {}
  ProfiledCallGraphNode *Source;
  ProfiledCallGraphNode *Target;
  uint64_t Weight;

  // The call destination is the only important data here,
  // allow to transparently unwrap into it.
  operator ProfiledCallGraphNode *() const { return Target; }
};

struct ProfiledCallGraphNode {

  // Sort edges by callee names only since all edges to be compared are from
  // same caller. Edge weights are not considered either because for the same
  // callee only the edge with the largest weight is added to the edge set.
  struct ProfiledCallGraphEdgeComparer {
    bool operator()(const ProfiledCallGraphEdge &L,
                    const ProfiledCallGraphEdge &R) const {
      return L.Target->Name < R.Target->Name;
    }
  };

  using iterator = std::set<ProfiledCallGraphEdge>::iterator;
  using const_iterator = std::set<ProfiledCallGraphEdge>::const_iterator;
  using edge = ProfiledCallGraphEdge;
  using edges = std::set<ProfiledCallGraphEdge, ProfiledCallGraphEdgeComparer>;

  ProfiledCallGraphNode(StringRef FName = StringRef()) : Name(FName) {}

  StringRef Name;
  edges Edges;
};

class ProfiledCallGraph {
public:
  using iterator = std::set<ProfiledCallGraphEdge>::iterator;

  // Constructor for non-CS profile.
  ProfiledCallGraph(SampleProfileMap &ProfileMap) {
    assert(!FunctionSamples::ProfileIsCS && "CS profile is not handled here");
    for (const auto &Samples : ProfileMap) {
      addProfiledCalls(Samples.second);
    }
  }

  // Constructor for CS profile.
  ProfiledCallGraph(SampleContextTracker &ContextTracker) {
    // BFS traverse the context profile trie to add call edges for calls shown
    // in context.
    std::queue<ContextTrieNode *> Queue;
    for (auto &Child : ContextTracker.getRootContext().getAllChildContext()) {
      ContextTrieNode *Callee = &Child.second;
      addProfiledFunction(ContextTracker.getFuncNameFor(Callee));
      Queue.push(Callee);
    }

    while (!Queue.empty()) {
      ContextTrieNode *Caller = Queue.front();
      Queue.pop();
      FunctionSamples *CallerSamples = Caller->getFunctionSamples();

      // Add calls for context.
      // Note that callsite target samples are completely ignored since they can
      // conflict with the context edges, which are formed by context
      // compression during profile generation, for cyclic SCCs. This may
      // further result in an SCC order incompatible with the purely
      // context-based one, which may in turn block context-based inlining.
      for (auto &Child : Caller->getAllChildContext()) {
        ContextTrieNode *Callee = &Child.second;
        addProfiledFunction(ContextTracker.getFuncNameFor(Callee));
        Queue.push(Callee);

        // Fetch edge weight from the profile.
        uint64_t Weight;
        FunctionSamples *CalleeSamples = Callee->getFunctionSamples();
        if (!CalleeSamples || !CallerSamples) {
          Weight = 0;
        } else {
          uint64_t CalleeEntryCount = CalleeSamples->getEntrySamples();
          uint64_t CallsiteCount = 0;
          LineLocation Callsite = Callee->getCallSiteLoc();
          if (auto CallTargets = CallerSamples->findCallTargetMapAt(Callsite)) {
            SampleRecord::CallTargetMap &TargetCounts = CallTargets.get();
            auto It = TargetCounts.find(CalleeSamples->getName());
            if (It != TargetCounts.end())
              CallsiteCount = It->second;
          }
          Weight = std::max(CallsiteCount, CalleeEntryCount);
        }

        addProfiledCall(ContextTracker.getFuncNameFor(Caller),
                        ContextTracker.getFuncNameFor(Callee), Weight);
      }
    }
  }

  iterator begin() { return Root.Edges.begin(); }
  iterator end() { return Root.Edges.end(); }
  ProfiledCallGraphNode *getEntryNode() { return &Root; }
  void addProfiledFunction(StringRef Name) {
    if (!ProfiledFunctions.count(Name)) {
      // Link to synthetic root to make sure every node is reachable
      // from root. This does not affect SCC order.
      ProfiledFunctions[Name] = ProfiledCallGraphNode(Name);
      Root.Edges.emplace(&Root, &ProfiledFunctions[Name], 0);
    }
  }

private:
  void addProfiledCall(StringRef CallerName, StringRef CalleeName,
                       uint64_t Weight = 0) {
    assert(ProfiledFunctions.count(CallerName));
    auto CalleeIt = ProfiledFunctions.find(CalleeName);
    if (CalleeIt == ProfiledFunctions.end())
      return;
    ProfiledCallGraphEdge Edge(&ProfiledFunctions[CallerName],
                               &CalleeIt->second, Weight);
    auto &Edges = ProfiledFunctions[CallerName].Edges;
    auto EdgeIt = Edges.find(Edge);
    if (EdgeIt == Edges.end()) {
      Edges.insert(Edge);
    } else if (EdgeIt->Weight < Edge.Weight) {
      // Replace existing call edges with same target but smaller weight.
      Edges.erase(EdgeIt);
      Edges.insert(Edge);
    }
  }

  void addProfiledCalls(const FunctionSamples &Samples) {
    addProfiledFunction(Samples.getFuncName());

    for (const auto &Sample : Samples.getBodySamples()) {
      for (const auto &Target : Sample.second.getCallTargets()) {
        addProfiledFunction(Target.first());
        addProfiledCall(Samples.getFuncName(), Target.first(), Target.second);
      }
    }

    for (const auto &CallsiteSamples : Samples.getCallsiteSamples()) {
      for (const auto &InlinedSamples : CallsiteSamples.second) {
        addProfiledFunction(InlinedSamples.first);
        addProfiledCall(Samples.getFuncName(), InlinedSamples.first,
                        InlinedSamples.second.getEntrySamples());
        addProfiledCalls(InlinedSamples.second);
      }
    }
  }

  ProfiledCallGraphNode Root;
  StringMap<ProfiledCallGraphNode> ProfiledFunctions;
};

} // end namespace sampleprof

template <> struct GraphTraits<ProfiledCallGraphNode *> {
  using NodeType = ProfiledCallGraphNode;
  using NodeRef = ProfiledCallGraphNode *;
  using EdgeType = NodeType::edge;
  using ChildIteratorType = NodeType::const_iterator;

  static NodeRef getEntryNode(NodeRef PCGN) { return PCGN; }
  static ChildIteratorType child_begin(NodeRef N) { return N->Edges.begin(); }
  static ChildIteratorType child_end(NodeRef N) { return N->Edges.end(); }
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
