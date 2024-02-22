//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StatsCollector.h"
#include "llvm/CAS/TreeSchema.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;
using namespace llvm::cas;

void StatsCollector::visitPOT(ExitOnError &ExitOnErr,
                              ArrayRef<ObjectProxy> TopLevels,
                              ArrayRef<POTItem> POT) {
  for (auto ID : TopLevels)
    Nodes[ID.getRef()].NumPaths = 1;

  // Visit POT in reverse (topological sort), computing Nodes and collecting
  // stats.
  for (const POTItem &Item : llvm::reverse(POT))
    visitPOTItem(ExitOnErr, Item);
}

void StatsCollector::visitPOTItem(ExitOnError &ExitOnErr, const POTItem &Item) {
  cas::ObjectRef ID = Item.ID;
  size_t NumPaths = Nodes.lookup(ID).NumPaths;
  ObjectProxy Object = ExitOnErr(CAS.getProxy(ID));

  auto updateChild = [&](ObjectRef Child) {
    ++Nodes[Child].NumParents;
    Nodes[Child].NumPaths += NumPaths;
  };

  size_t NumParents = Nodes.lookup(ID).NumParents;
  TreeSchema Schema(CAS);
  if (Schema.isNode(Object)) {
    auto &Info = Stats["builtin:tree"];
    ++Info.Count;
    TreeProxy Tree = ExitOnErr(Schema.load(Object));
    Info.NumChildren += Tree.size();
    Info.NumParents += NumParents;
    Info.NumPaths += NumPaths;
    if (!Tree.size())
      Totals.NumPaths += NumPaths; // Count paths to leafs.

    ExitOnErr(Tree.forEachEntry([&](const NamedTreeEntry &Entry) {
      // FIXME: This is copied out of BuiltinCAS.cpp's makeTree.
      Info.DataSize += sizeof(uint64_t) + sizeof(uint32_t) +
                       alignTo(Entry.getName().size() + 1, Align(4));
      updateChild(Entry.getRef());
      return Error::success();
    }));
    return;
  }

  auto addNodeStats = [&](ObjectKindInfo &Info) {
    ++Info.Count;
    Info.NumChildren += Object.getNumReferences();
    Info.NumParents += NumParents;
    Info.NumPaths += NumPaths;
    Info.DataSize += Object.getData().size();
    if (!Object.getNumReferences())
      Totals.NumPaths += NumPaths; // Count paths to leafs.

    ExitOnErr(Object.forEachReference([&](ObjectRef Child) {
      updateChild(Child);
      return Error::success();
    }));
  };

  // Handle nodes not in the schema.
  if (!Item.Schema) {
    addNodeStats(Stats["builtin:node"]);
    return;
  }

  for (auto &S : Schemas) {
    if (Item.Schema != S.first)
      continue;
    S.second(ExitOnErr, addNodeStats, Object);
    return;
  }
  llvm_unreachable("schema not found");
}

void StatsCollector::visitPOTItemMCCASV1(
    ExitOnError &ExitOnErr, llvm::mccasformats::v1::MCSchema &Schema,
    function_ref<void(ObjectKindInfo &Info)> addNodeStats,
    cas::ObjectProxy Node) {
  auto Object = ExitOnErr(Schema.get(Node.getRef()));
  addNodeStats(Stats[Object.getKindString()]);

  if (Object.getNumReferences() == 0 &&
      Object.getData().size() <
          (Object.getID().getHash().size() + sizeof(void *)))
    ++NumTinyObjects;

  if (Object.getKindString() ==
      llvm::mccasformats::v1::SectionRef::KindString) {
    if (!Object.getData().empty())
      SecRefSize += Object.getData().size();
  }
  if (Object.getKindString() == llvm::mccasformats::v1::AtomRef::KindString) {
    if (!Object.getData().empty())
      AtomRefSize += Object.getData().size();
  }
}

void StatsCollector::printToOuts(ArrayRef<ObjectProxy> TopLevels,
                                 raw_ostream &StatOS) {

  SmallVector<StringRef> Kinds;
  auto addToTotal = [&Totals = this->Totals](ObjectKindInfo Info) {
    Totals.Count += Info.Count;
    Totals.NumChildren += Info.NumChildren;
    Totals.NumParents += Info.NumParents;
    Totals.DataSize += Info.DataSize;
  };
  for (auto &I : Stats) {
    Kinds.push_back(I.first());
    addToTotal(I.second);
  }
  llvm::sort(Kinds);

  CASID FirstID = TopLevels.front().getID();
  size_t NumHashBytes = FirstID.getHash().size();
  if (ObjectStatsFormat == FormatType::Pretty) {
    StatOS
        << "  => Note: 'Parents' counts incoming edges\n"
        << "  => Note: 'Children' counts outgoing edges (to sub-objects)\n"
        << "  => Note: HashSize = " << NumHashBytes << "B\n"
        << "  => Note: PtrSize  = " << sizeof(void *) << "B\n"
        << "  => Note: Cost     = Count*HashSize + PtrSize*Children + Data\n";
  }
  StringLiteral HeaderFormatPretty =
      "{0,-22} {1,+10} {2,+7} {3,+10} {4,+7} {5,+10} "
      "{6,+7} {7,+10} {8,+7} {9,+10} {10,+7}\n";
  StringLiteral FormatPretty =
      "{0,-22} {1,+10:N} {2,+7:P} {3,+10:N} {4,+7:P} {5,+10:N} "
      "{6,+7:P} {7,+10:N} {8,+7:P} {9,+10:N} {10,+7:P}\n";
  StringLiteral FormatCSV = "{0}, {1}, {3}, {5}, {7}, {9}\n";

  StringLiteral HeaderFormat =
      ObjectStatsFormat == FormatType::Pretty ? HeaderFormatPretty : FormatCSV;
  StringLiteral Format =
      ObjectStatsFormat == FormatType::Pretty ? FormatPretty : FormatCSV;

  StatOS << llvm::formatv(HeaderFormat.begin(), "Kind", "Count", "", "Parents",
                          "", "Children", "", "Data (B)", "", "Cost (B)", "");
  if (ObjectStatsFormat == FormatType::Pretty) {
    StatOS << llvm::formatv(HeaderFormat.begin(), "====", "=====", "",
                            "=======", "", "========", "", "========", "",
                            "========", "");
  }

  auto printInfo = [&](StringRef Kind, ObjectKindInfo Info) {
    if (!Info.Count)
      return;
    auto getPercent = [](double N, double D) { return D ? N / D : 0.0; };
    size_t Size = Info.getTotalSize(NumHashBytes);
    StatOS << llvm::formatv(
        Format.begin(), Kind, Info.Count, getPercent(Info.Count, Totals.Count),
        Info.NumParents, getPercent(Info.NumParents, Totals.NumParents),
        Info.NumChildren, getPercent(Info.NumChildren, Totals.NumChildren),
        Info.DataSize, getPercent(Info.DataSize, Totals.DataSize), Size,
        getPercent(Size, Totals.getTotalSize(NumHashBytes)));
  };
  for (StringRef Kind : Kinds)
    printInfo(Kind, Stats.lookup(Kind));
  printInfo("TOTAL", Totals);

  StringLiteral OtherStatsPretty = "{0,-22} {1,+10}\n";
  StringLiteral OtherStatsCSV = "{0}, {1}\n";

  StringLiteral OtherStats = ObjectStatsFormat == FormatType::Pretty
                                 ? OtherStatsPretty
                                 : OtherStatsCSV;
  // Other stats.
  bool HasPrinted = false;
  auto printIfNotZero = [&](StringRef Name, size_t Num) {
    if (!Num)
      return;
    if (!HasPrinted)
      StatOS << "\n";
    StatOS << llvm::formatv(OtherStats.begin(), Name, Num);
    HasPrinted = true;
  };

  printIfNotZero("num-generated-names", GeneratedNames.size());
  printIfNotZero("num-section-names", SectionNames.size());
  printIfNotZero("num-symbol-names",
                 SymbolNames.size() + UndefinedSymbols.size());
  printIfNotZero("num-undefined-symbols", UndefinedSymbols.size());
  printIfNotZero("num-anonymous-symbols", NumAnonymousSymbols);
  printIfNotZero("num-template-symbols", NumTemplateSymbols);
  printIfNotZero("num-template-targets", NumTemplateTargets);
  printIfNotZero("num-zero-fill-blocks", NumZeroFillBlocks);
  printIfNotZero("num-1-target-blocks", Num2TargetBlocks);
  printIfNotZero("num-2-target-blocks", Num1TargetBlocks);
  printIfNotZero("num-tiny-objects", NumTinyObjects);
  printIfNotZero("sec-ref-size", SecRefSize);
  printIfNotZero("atom-ref-size", AtomRefSize);
}
