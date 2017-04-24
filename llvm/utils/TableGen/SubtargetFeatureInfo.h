//===- SubtargetFeatureInfo.h - Helpers for subtarget features ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTIL_TABLEGEN_SUBTARGETFEATUREINFO_H
#define LLVM_UTIL_TABLEGEN_SUBTARGETFEATUREINFO_H

#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

#include <map>
#include <string>
#include <vector>

namespace llvm {
class Record;
class RecordKeeper;

/// Helper class for storing information on a subtarget feature which
/// participates in instruction matching.
struct SubtargetFeatureInfo {
  /// \brief The predicate record for this feature.
  Record *TheDef;

  /// \brief An unique index assigned to represent this feature.
  uint64_t Index;

  SubtargetFeatureInfo(Record *D, uint64_t Idx) : TheDef(D), Index(Idx) {}

  /// \brief The name of the enumerated constant identifying this feature.
  std::string getEnumName() const {
    return "Feature_" + TheDef->getName().str();
  }

  /// \brief The name of the enumerated constant identifying the bitnumber for
  /// this feature.
  std::string getEnumBitName() const {
    return "Feature_" + TheDef->getName().str() + "Bit";
  }

  void dump() const;
  static std::vector<std::pair<Record *, SubtargetFeatureInfo>>
  getAll(const RecordKeeper &Records);

  /// Emit the subtarget feature flag definitions.
  ///
  /// This version emits the bit value for the feature and is therefore limited
  /// to 64 feature bits.
  static void emitSubtargetFeatureFlagEnumeration(
      std::map<Record *, SubtargetFeatureInfo, LessRecordByID>
          &SubtargetFeatures,
      raw_ostream &OS);

  /// Emit the subtarget feature flag definitions.
  ///
  /// This version emits the bit index for the feature and can therefore support
  /// more than 64 feature bits.
  static void emitSubtargetFeatureBitEnumeration(
      std::map<Record *, SubtargetFeatureInfo, LessRecordByID>
          &SubtargetFeatures,
      raw_ostream &OS);

  static void emitNameTable(std::map<Record *, SubtargetFeatureInfo,
                                     LessRecordByID> &SubtargetFeatures,
                            raw_ostream &OS);

  /// Emit the function to compute the list of available features given a
  /// subtarget.
  ///
  /// This version is used for subtarget features defined using Predicate<>
  /// and supports more than 64 feature bits.
  ///
  /// \param TargetName The name of the target as used in class prefixes (e.g.
  ///                   <TargetName>Subtarget)
  /// \param ClassName  The name of the class (without the <Target> prefix)
  ///                   that will contain the generated functions.
  /// \param FuncName   The name of the function to emit.
  /// \param SubtargetFeatures A map of TableGen records to the
  ///                          SubtargetFeatureInfo equivalent.
  static void emitComputeAvailableFeatures(
      StringRef TargetName, StringRef ClassName, StringRef FuncName,
      std::map<Record *, SubtargetFeatureInfo, LessRecordByID>
          &SubtargetFeatures,
      raw_ostream &OS);

  /// Emit the function to compute the list of available features given a
  /// subtarget.
  ///
  /// This version is used for subtarget features defined using
  /// AssemblerPredicate<> and supports up to 64 feature bits.
  ///
  /// \param TargetName The name of the target as used in class prefixes (e.g.
  ///                   <TargetName>Subtarget)
  /// \param ClassName  The name of the class (without the <Target> prefix)
  ///                   that will contain the generated functions.
  /// \param FuncName   The name of the function to emit.
  /// \param SubtargetFeatures A map of TableGen records to the
  ///                          SubtargetFeatureInfo equivalent.
  static void emitComputeAssemblerAvailableFeatures(
      StringRef TargetName, StringRef ClassName, StringRef FuncName,
      std::map<Record *, SubtargetFeatureInfo, LessRecordByID>
          &SubtargetFeatures,
      raw_ostream &OS);
};
} // end namespace llvm

#endif // LLVM_UTIL_TABLEGEN_SUBTARGETFEATUREINFO_H
