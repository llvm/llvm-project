//===--- BasicGOTAndStubsBuilder.h - Generic GOT/Stub creation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A base for simple GOT and stub creation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_EXECUTIONENGINE_JITLINK_BASICGOTANDSTUBSBUILDER_H
#define LLVM_LIB_EXECUTIONENGINE_JITLINK_BASICGOTANDSTUBSBUILDER_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

namespace llvm {
namespace jitlink {

template <typename BuilderImpl> class BasicGOTAndStubsBuilder {
public:
  BasicGOTAndStubsBuilder(AtomGraph &G) : G(G) {}

  void run() {
    // We're going to be adding new atoms, but we don't want to iterate over
    // the newly added ones, so just copy the existing atoms out.
    std::vector<DefinedAtom *> DAs(G.defined_atoms().begin(),
                                   G.defined_atoms().end());

    for (auto *DA : DAs)
      for (auto &E : DA->edges())
        if (impl().isGOTEdge(E))
          impl().fixGOTEdge(E, getGOTEntryAtom(E.getTarget()));
        else if (impl().isExternalBranchEdge(E))
          impl().fixExternalBranchEdge(E, getStubAtom(E.getTarget()));
  }

protected:
  Atom &getGOTEntryAtom(Atom &Target) {
    assert(Target.hasName() && "GOT edge cannot point to anonymous target");

    auto GOTEntryI = GOTEntries.find(Target.getName());

    // Build the entry if it doesn't exist.
    if (GOTEntryI == GOTEntries.end()) {
      auto &GOTEntry = impl().createGOTEntry(Target);
      GOTEntryI =
          GOTEntries.insert(std::make_pair(Target.getName(), &GOTEntry)).first;
    }

    assert(GOTEntryI != GOTEntries.end() && "Could not get GOT entry atom");
    return *GOTEntryI->second;
  }

  Atom &getStubAtom(Atom &Target) {
    assert(Target.hasName() &&
           "External branch edge can not point to an anonymous target");
    auto StubI = Stubs.find(Target.getName());

    if (StubI == Stubs.end()) {
      auto &StubAtom = impl().createStub(Target);
      StubI = Stubs.insert(std::make_pair(Target.getName(), &StubAtom)).first;
    }

    assert(StubI != Stubs.end() && "Count not get stub atom");
    return *StubI->second;
  }

  AtomGraph &G;

private:
  BuilderImpl &impl() { return static_cast<BuilderImpl &>(*this); }

  DenseMap<StringRef, DefinedAtom *> GOTEntries;
  DenseMap<StringRef, DefinedAtom *> Stubs;
};

} // end namespace jitlink
} // end namespace llvm

#endif // LLVM_LIB_EXECUTIONENGINE_JITLINK_BASICGOTANDSTUBSBUILDER_H
