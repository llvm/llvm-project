//===- Analyze.cpp - PDB analysis functions ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Analyze.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/LazyRandomTypeCollection.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <list>

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

static StringRef getLeafTypeName(TypeLeafKind LT) {
  switch (LT) {
#define TYPE_RECORD(ename, value, name)                                        \
  case ename:                                                                  \
    return #name;
#include "llvm/DebugInfo/CodeView/CodeViewTypes.def"
  default:
    break;
  }
  return "UnknownLeaf";
}

namespace {
struct HashLookupVisitor : public TypeVisitorCallbacks {
  struct Entry {
    TypeIndex TI;
    CVType Record;
  };

  explicit HashLookupVisitor(TpiStream &Tpi) : Tpi(Tpi) {}

  Error visitTypeBegin(CVType &Record) override {
    uint32_t H = Tpi.getHashValues()[I];
    Record.Hash = H;
    TypeIndex TI(I + TypeIndex::FirstNonSimpleIndex);
    Lookup[H].push_back(Entry{TI, Record});
    ++I;
    return Error::success();
  }

  uint32_t I = 0;
  DenseMap<uint32_t, std::list<Entry>> Lookup;
  TpiStream &Tpi;
};
}

AnalysisStyle::AnalysisStyle(PDBFile &File) : File(File) {}

Error AnalysisStyle::dump() {
  auto Tpi = File.getPDBTpiStream();
  if (!Tpi)
    return Tpi.takeError();

  HashLookupVisitor Hasher(*Tpi);

  uint32_t RecordCount = Tpi->getNumTypeRecords();
  auto Offsets = Tpi->getTypeIndexOffsets();
  auto Types = llvm::make_unique<LazyRandomTypeCollection>(
      Tpi->typeArray(), RecordCount, Offsets);

  if (auto EC = codeview::visitTypeStream(*Types, Hasher))
    return EC;

  auto &Adjusters = Tpi->getHashAdjusters();
  DenseSet<uint32_t> AdjusterSet;
  for (const auto &Adj : Adjusters) {
    assert(AdjusterSet.find(Adj.second) == AdjusterSet.end());
    AdjusterSet.insert(Adj.second);
  }

  uint32_t Count = 0;
  outs() << "Searching for hash collisions\n";
  for (const auto &H : Hasher.Lookup) {
    if (H.second.size() <= 1)
      continue;
    ++Count;
    outs() << formatv("Hash: {0}, Count: {1} records\n", H.first,
                      H.second.size());
    for (const auto &R : H.second) {
      auto Iter = AdjusterSet.find(R.TI.getIndex());
      StringRef Prefix;
      if (Iter != AdjusterSet.end()) {
        Prefix = "[HEAD]";
        AdjusterSet.erase(Iter);
      }
      StringRef LeafName = getLeafTypeName(R.Record.Type);
      uint32_t TI = R.TI.getIndex();
      StringRef TypeName = Types->getTypeName(R.TI);
      outs() << formatv("{0,-6} {1} ({2:x}) {3}\n", Prefix, LeafName, TI,
                        TypeName);
    }
  }

  outs() << "\n";
  outs() << "Dumping hash adjustment chains\n";
  for (const auto &A : Tpi->getHashAdjusters()) {
    TypeIndex TI(A.second);
    StringRef TypeName = Types->getTypeName(TI);
    const CVType &HeadRecord = Types->getType(TI);
    assert(HeadRecord.Hash.hasValue());

    auto CollisionsIter = Hasher.Lookup.find(*HeadRecord.Hash);
    if (CollisionsIter == Hasher.Lookup.end())
      continue;

    const auto &Collisions = CollisionsIter->second;
    outs() << TypeName << "\n";
    outs() << formatv("    [HEAD] {0:x} {1} {2}\n", A.second,
                      getLeafTypeName(HeadRecord.Type), TypeName);
    for (const auto &Chain : Collisions) {
      if (Chain.TI == TI)
        continue;
      const CVType &TailRecord = Types->getType(Chain.TI);
      outs() << formatv("           {0:x} {1} {2}\n", Chain.TI.getIndex(),
                        getLeafTypeName(TailRecord.Type),
                        Types->getTypeName(Chain.TI));
    }
  }
  outs() << formatv("There are {0} orphaned hash adjusters\n",
                    AdjusterSet.size());
  for (const auto &Adj : AdjusterSet) {
    outs() << formatv("    {0}\n", Adj);
  }

  uint32_t DistinctHashValues = Hasher.Lookup.size();
  outs() << formatv("{0}/{1} hash collisions", Count, DistinctHashValues);
  return Error::success();
}
