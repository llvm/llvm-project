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
  const RecordKeeper &Records;
  std::string Target;

public:
  TargetFeaturesEmitter(const RecordKeeper &R);
  static void printFeatureMask(raw_ostream &OS,
                               ArrayRef<const Record *> FeatureList,
                               const FeatureMapTy &FeatureMap);
  FeatureMapTy enumeration(raw_ostream &OS);
  void printFeatureKeyValues(raw_ostream &OS, const FeatureMapTy &FeatureMap);
  void printCPUKeyValues(raw_ostream &OS, const FeatureMapTy &FeatureMap);
  virtual void run(raw_ostream &O);
  virtual ~TargetFeaturesEmitter() {};
};
} // namespace llvm
#endif
