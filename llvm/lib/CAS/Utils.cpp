//===- Utils.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/Utils.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/CAS/CASDB.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"

using namespace llvm;
using namespace llvm::cas;

Error cas::walkFileTreeRecursively(
    CASDB &CAS, const TreeHandle &Root,
    function_ref<Error(const NamedTreeEntry &, Optional<TreeProxy>)> Callback) {
  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);
  SmallString<128> PathStorage;
  SmallVector<NamedTreeEntry> Stack;
  Stack.emplace_back(CAS.getReference(Root), TreeEntry::Tree, "/");

  while (!Stack.empty()) {
    if (Stack.back().getKind() != TreeEntry::Tree) {
      if (Error E = Callback(Stack.pop_back_val(), None))
        return E;
      continue;
    }

    NamedTreeEntry Parent = Stack.pop_back_val();
    Expected<TreeProxy> ExpTree = CAS.loadTree(Parent.getRef());
    if (Error E = ExpTree.takeError())
      return E;
    TreeProxy Tree = *ExpTree;
    if (Error E = Callback(Parent, Tree))
      return E;
    for (int I = Tree.size(), E = 0; I != E; --I) {
      Optional<NamedTreeEntry> Child = Tree.get(I - 1);
      assert(Child && "Expected no corruption");

      SmallString<128> PathStorage = Parent.getName();
      sys::path::append(PathStorage, sys::path::Style::posix, Child->getName());
      Stack.emplace_back(Child->getRef(), Child->getKind(),
                         Saver.save(StringRef(PathStorage)));
    }
  }

  return Error::success();
}

static void printTreeEntryKind(raw_ostream &OS, TreeEntry::EntryKind Kind) {
  switch (Kind) {
  case TreeEntry::Regular:
    OS << "file";
    break;
  case TreeEntry::Executable:
    OS << "exec";
    break;
  case TreeEntry::Symlink:
    OS << "syml";
    break;
  case TreeEntry::Tree:
    OS << "tree";
    break;
  }
}

void cas::NamedTreeEntry::print(raw_ostream &OS, CASDB &CAS) const {
  printTreeEntryKind(OS, getKind());
  OS << " " << CAS.getObjectID(getRef()) << " " << Name;
  if (getKind() == TreeEntry::Tree)
    OS << "/";
  if (getKind() == TreeEntry::Symlink) {
    AnyObjectHandle Target = cantFail(CAS.loadObject(getRef()));
    OS << " -> ";
    CAS.readData(Target.get<NodeHandle>(), OS);
  }
  OS << "\n";
}
