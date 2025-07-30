//===-- llvm/IR/Mangler.h - Self-contained name mangler ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unified name mangler for various backends.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_MANGLER_H
#define LLVM_IR_MANGLER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class DataLayout;
class GlobalValue;
template <typename T> class SmallVectorImpl;
class Triple;
class Twine;
class raw_ostream;

// TODO: The weird assignment of HybridPatchableTargetSuffix below is a
// temporary workaround for a linker failure that is only hit when compiling
// llvm for arm64ec on windows. The description and context of the issue is at
// https://github.com/llvm/llvm-project/issues/143575.
// An upstream MSVC bug is filed at
// https://developercommunity.visualstudio.com/t/MSVC-Linker-Issue-When-Cross-
// Compiling-L/10920141.
constexpr char HybridPatchableTargetSuffixArr[] = "$hp_target";
constexpr std::string_view HybridPatchableTargetSuffix =
    HybridPatchableTargetSuffixArr;

class Mangler {
  /// We need to give global values the same name every time they are mangled.
  /// This keeps track of the number we give to anonymous ones.
  mutable DenseMap<const GlobalValue*, unsigned> AnonGlobalIDs;

public:
  /// Print the appropriate prefix and the specified global variable's name.
  /// If the global variable doesn't have a name, this fills in a unique name
  /// for the global.
  LLVM_ABI void getNameWithPrefix(raw_ostream &OS, const GlobalValue *GV,
                                  bool CannotUsePrivateLabel) const;
  LLVM_ABI void getNameWithPrefix(SmallVectorImpl<char> &OutName,
                                  const GlobalValue *GV,
                                  bool CannotUsePrivateLabel) const;

  /// Print the appropriate prefix and the specified name as the global variable
  /// name. GVName must not be empty.
  LLVM_ABI static void getNameWithPrefix(raw_ostream &OS, const Twine &GVName,
                                         const DataLayout &DL);
  LLVM_ABI static void getNameWithPrefix(SmallVectorImpl<char> &OutName,
                                         const Twine &GVName,
                                         const DataLayout &DL);
};

LLVM_ABI void emitLinkerFlagsForGlobalCOFF(raw_ostream &OS,
                                           const GlobalValue *GV,
                                           const Triple &TT, Mangler &Mangler);

LLVM_ABI void emitLinkerFlagsForUsedCOFF(raw_ostream &OS, const GlobalValue *GV,
                                         const Triple &T, Mangler &M);

/// Returns the ARM64EC mangled function name unless the input is already
/// mangled.
LLVM_ABI std::optional<std::string>
getArm64ECMangledFunctionName(StringRef Name);

/// Returns the ARM64EC demangled function name, unless the input is not
/// mangled.
LLVM_ABI std::optional<std::string>
getArm64ECDemangledFunctionName(StringRef Name);

/// Check if an ARM64EC function name is mangled.
bool inline isArm64ECMangledFunctionName(StringRef Name) {
  return Name[0] == '#' ||
         (Name[0] == '?' && Name.find("@$$h") != StringRef::npos);
}

} // End llvm namespace

#endif
