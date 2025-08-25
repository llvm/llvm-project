//===- llvm/MC/CAS/MCCASReader.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_CAS_MCCASREADER_H
#define LLVM_MC_CAS_MCCASREADER_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Error.h"

namespace llvm {

class Triple;

namespace mccasformats {
namespace reader {

/// Only valid to be used with the same \p CASMCReader instance it came
/// from.
struct CASSectionRef {
  uint64_t Idx;
};

/// Only valid to be used with the same \p CASMCReader instance it came
/// from.
struct CASFragmentRef {
  uint64_t Idx;
};

/// Only valid to be used with the same \p CASMCReader instance it came
/// from.
struct CASSymbolRef {
  uint64_t Idx;
};

class MCSectionCAS : public MCSection {
public:
  MCSectionCAS(StringRef Name, SectionKind K);
  virtual ~MCSectionCAS() {}
};

class MCFragmentCAS : public MCFragment {
public:
  MCFragmentCAS(FragmentType Kind, bool HasInstructions);

private:
};

class MCFixupCAS : public MCFixup {};

class MCSymbolCAS : public MCSymbol {};

class CASMCReader {
public:
  virtual ~CASMCReader() = default;

  virtual Triple getTargetTriple() const = 0;

  /// Get all symbols in the translation unit.
  /// FIXME: Nestedv1 schema doesn't provide all the symbols that can be
  /// referenced. More can be discovered via fixups.
  virtual Error forEachSymbol(
      function_ref<Error(CASSymbolRef, MCSymbolCAS)> Callback) const = 0;

  virtual Expected<MCSectionCAS> materialize(CASSectionRef Ref) const = 0;
  virtual Expected<MCFragmentCAS> materialize(CASFragmentRef Ref) const = 0;
  virtual Expected<MCSymbolCAS> materialize(CASSymbolRef Ref) const = 0;

  virtual Error
  materializeFixups(CASFragmentRef Ref,
                    function_ref<Error(const MCFixupCAS &)> Callback) const = 0;

  MCSectionCAS *createSection(StringRef Name, SectionKind K);

protected:
  CASMCReader() = default;

private:

};

} // namespace reader
} // namespace mccasformats
} // namespace llvm

namespace llvm {

template <> struct DenseMapInfo<mccasformats::reader::CASSectionRef> {
  static mccasformats::reader::CASSectionRef getEmptyKey() {
    return mccasformats::reader::CASSectionRef{
        DenseMapInfo<uint64_t>::getEmptyKey()};
  }

  static mccasformats::reader::CASSectionRef getTombstoneKey() {
    return mccasformats::reader::CASSectionRef{
        DenseMapInfo<uint64_t>::getTombstoneKey()};
  }

  static unsigned getHashValue(mccasformats::reader::CASSectionRef ID) {
    return DenseMapInfo<uint64_t>::getHashValue(ID.Idx);
  }

  static bool isEqual(mccasformats::reader::CASSectionRef LHS,
                      mccasformats::reader::CASSectionRef RHS) {
    return LHS.Idx == RHS.Idx;
  }
};

template <> struct DenseMapInfo<mccasformats::reader::CASFragmentRef> {
  static mccasformats::reader::CASFragmentRef getEmptyKey() {
    return mccasformats::reader::CASFragmentRef{
        DenseMapInfo<uint64_t>::getEmptyKey()};
  }

  static mccasformats::reader::CASFragmentRef getTombstoneKey() {
    return mccasformats::reader::CASFragmentRef{
        DenseMapInfo<uint64_t>::getTombstoneKey()};
  }

  static unsigned getHashValue(mccasformats::reader::CASFragmentRef ID) {
    return DenseMapInfo<uint64_t>::getHashValue(ID.Idx);
  }

  static bool isEqual(mccasformats::reader::CASFragmentRef LHS,
                      mccasformats::reader::CASFragmentRef RHS) {
    return LHS.Idx == RHS.Idx;
  }
};

template <> struct DenseMapInfo<mccasformats::reader::CASSymbolRef> {
  static mccasformats::reader::CASSymbolRef getEmptyKey() {
    return mccasformats::reader::CASSymbolRef{
        DenseMapInfo<uint64_t>::getEmptyKey()};
  }

  static mccasformats::reader::CASSymbolRef getTombstoneKey() {
    return mccasformats::reader::CASSymbolRef{
        DenseMapInfo<uint64_t>::getTombstoneKey()};
  }

  static unsigned getHashValue(mccasformats::reader::CASSymbolRef ID) {
    return DenseMapInfo<uint64_t>::getHashValue(ID.Idx);
  }

  static bool isEqual(mccasformats::reader::CASSymbolRef LHS,
                      mccasformats::reader::CASSymbolRef RHS) {
    return LHS.Idx == RHS.Idx;
  }
};

} // namespace llvm

#endif // LLVM_MC_CAS_MCCASREADER_H
