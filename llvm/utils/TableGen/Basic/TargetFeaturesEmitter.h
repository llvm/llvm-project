//===- TargetFeaturesEmitter.h- Generate CPU Target features ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the TargetFeaturesEmitter class, which is used to export
// CPU target features and CPU subtypes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_BASIC_EMITTARGETFEATURE_H
#define LLVM_UTILS_TABLEGEN_BASIC_EMITTARGETFEATURE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include <array>
#include <cstdint>
#include <vector>

namespace llvm {
/// Sorting predicate to sort record pointers by their
/// FieldName field.
struct LessRecordFieldFieldName {
  bool operator()(const Record *Rec1, const Record *Rec2) const {
    return Rec1->getValueAsString("FieldName") <
           Rec2->getValueAsString("FieldName");
  }
};

using FeatureMapTy = DenseMap<const Record *, unsigned>;

class TargetFeaturesEmitter {
protected:
  using FeatureMask = std::array<uint64_t, MAX_SUBTARGET_WORDS>;
  using FeatureMaskTable = std::vector<FeatureMask>;

  const RecordKeeper &Records;
  std::string Target;

  static FeatureMask getFeatureMask(ArrayRef<const Record *> FeatureList,
                                    const FeatureMapTy &FeatureMap);
  static unsigned getFeatureMaskIndex(ArrayRef<const Record *> FeatureList,
                                      const FeatureMapTy &FeatureMap,
                                      FeatureMaskTable &FeatureMasks);
  static void printFeatureMask(raw_ostream &OS, const FeatureMask &Mask);
  static void printFeatureMaskTable(raw_ostream &OS, StringRef TableName,
                                    ArrayRef<FeatureMask> FeatureMasks);

public:
  TargetFeaturesEmitter(const RecordKeeper &R);
  FeatureMapTy enumeration(raw_ostream &OS);
  void printFeatureKeyValues(raw_ostream &OS, const FeatureMapTy &FeatureMap,
                             FeatureMaskTable &FeatureMasks);
  void printCPUKeyValues(raw_ostream &OS, const FeatureMapTy &FeatureMap,
                         FeatureMaskTable &FeatureMasks);
  virtual void run(raw_ostream &O);
  virtual ~TargetFeaturesEmitter() = default;
};
} // namespace llvm
#endif
