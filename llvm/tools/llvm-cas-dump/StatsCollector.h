//===-----------------------------------------------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_CAS_DUMP_STATSCOLLECTOR_H
#define LLVM_TOOLS_LLVM_CAS_DUMP_STATSCOLLECTOR_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/CAS/CASNodeSchema.h"
#include "llvm/CAS/CASReference.h"
#include "llvm/MCCAS/MCCASObjectV1.h"

namespace llvm {
namespace cas {

struct NodeInfo {
  size_t NumPaths = 0;
  size_t NumParents = 0;
  bool Done = false;
};

struct POTItem {
  ObjectRef ID;
  const NodeSchema *Schema = nullptr;
};

struct ObjectKindInfo {
  size_t Count = 0;
  size_t NumPaths = 0;
  size_t NumChildren = 0;
  size_t NumParents = 0;
  size_t DataSize = 0;

  size_t getTotalSize(size_t NumHashBytes) const {
    return Count * NumHashBytes + NumChildren * sizeof(void *) + DataSize;
  }
};

struct StatsCollector {
  ObjectStore &CAS;

  enum FormatType { Pretty, CSV };

  FormatType ObjectStatsFormat;

  using POTItemHandler = unique_function<void(
      ExitOnError &, function_ref<void(ObjectKindInfo &)>, cas::ObjectProxy)>;

  // FIXME: Utilize \p SchemaPool.
  llvm::mccasformats::v1::MCSchema MCCASV1Schema;
  SmallVector<std::pair<const NodeSchema *, POTItemHandler>> Schemas;

  StatsCollector(ObjectStore &CAS, FormatType Format)
      : CAS(CAS), ObjectStatsFormat(Format), MCCASV1Schema(CAS) {
    Schemas.push_back(std::make_pair(
        &MCCASV1Schema,
        [&](ExitOnError &ExitOnErr,
            function_ref<void(ObjectKindInfo & Info)> addNodeStats,
            cas::ObjectProxy Node) {
          visitPOTItemMCCASV1(ExitOnErr, MCCASV1Schema, addNodeStats, Node);
        }));
  }

  DenseMap<ObjectRef, NodeInfo> Nodes;
  StringMap<ObjectKindInfo> Stats;
  ObjectKindInfo Totals;
  DenseSet<StringRef> GeneratedNames;
  DenseSet<cas::ObjectRef> SectionNames;
  DenseSet<cas::ObjectRef> SymbolNames;
  DenseSet<cas::ObjectRef> UndefinedSymbols;
  DenseSet<cas::ObjectRef> ContentBlobs;
  size_t NumAnonymousSymbols = 0;
  size_t NumTemplateSymbols = 0;
  size_t NumTemplateTargets = 0;
  size_t NumZeroFillBlocks = 0;
  size_t Num1TargetBlocks = 0;
  size_t Num2TargetBlocks = 0;
  size_t NumTinyObjects = 0;
  size_t SecRefSize = 0;
  size_t AtomRefSize = 0;

  void visitPOT(ExitOnError &ExitOnErr, ArrayRef<ObjectProxy> TopLevels,
                ArrayRef<POTItem> POT);
  void visitPOTItem(ExitOnError &ExitOnErr, const POTItem &Item);
  void
  visitPOTItemMCCASV1(ExitOnError &ExitOnErr,
                      llvm::mccasformats::v1::MCSchema &Schema,
                      function_ref<void(ObjectKindInfo &Info)> addNodeStats,
                      cas::ObjectProxy Node);
  void printToOuts(ArrayRef<ObjectProxy> TopLevels, raw_ostream &StatOS);
};
} // namespace cas
} // namespace llvm
#endif