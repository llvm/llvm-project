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
#include "llvm/CAS/TreeSchema.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"

using namespace llvm;
using namespace llvm::cas;

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
  OS << " " << CAS.getID(getRef()) << " " << Name;
  if (getKind() == TreeEntry::Tree)
    OS << "/";
  if (getKind() == TreeEntry::Symlink) {
    ObjectHandle Target = cantFail(CAS.load(getRef()));
    OS << " -> ";
    CAS.readData(Target, OS);
  }
  OS << "\n";
}
