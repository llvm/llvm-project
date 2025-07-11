//===- CovMap.h - ObjectYAML Interface for coverage map ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// - llvm::coverage::yaml
//
//   Describes binary file formats and YAML structures of coverage map.
//
// - llvm::yaml
//
//   Attachments for YAMLTraits.
//
// - llvm::covmap
//
//   Provides YAML encoder for coverage map.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECTYAML_COVMAP_H
#define LLVM_OBJECTYAML_COVMAP_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ObjectYAML/ELFYAML.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/YAMLTraits.h"
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace llvm::coverage::yaml {

/// Base Counter, corresponding to coverage::Counter.
struct CounterTy {
  enum TagTy : uint8_t {
    Zero = 0,
    Ref,
    Sub,
    Add,
  };

  TagTy Tag;
  uint64_t Val;

  virtual ~CounterTy() {}

  virtual void mapping(llvm::yaml::IO &IO);

  void encode(raw_ostream &OS) const;
};

/// Holds a pair of both hands but doesn't hold ops(add or sub).
/// Ops is stored in CounterTy::Tag.
using ExpressionTy = std::array<CounterTy, 2>;

/// {True, False}
using BranchTy = std::array<CounterTy, 2>;

/// {ID, TrueID, FalseID}
/// Note: This has +1 offset unlike mcdc::ConditionID.
using MCDCBranchTy = std::array<uint16_t, 3>;

struct DecisionTy {
  uint64_t BIdx; ///< Bitmap index
  uint64_t NC;   ///< NumConds

  void mapping(llvm::yaml::IO &IO);

  void encode(raw_ostream &OS) const;
};

/// {LineStart, ColumnStart, LineEnd, ColumnEnd}
using LocTy = std::array<uint64_t, 4>;

/// Region record.
/// CounterTy is enhanced if Tag is Zero and Val is not zero.
struct RecTy : CounterTy {
  enum ExtTagTy : uint8_t {
    Skip = 2,
    Branch = 4,
    Decision = 5,
    MCDCBranch = 6,
  };

  /// This is optional in detailed view.
  std::optional<ExtTagTy> ExtTag;

  // Options for extensions.
  std::optional<uint64_t> Expansion; ///< Doesn't have ExtTag.
  std::optional<BranchTy> BranchOpt; ///< Optionally has MCDC.
  std::optional<MCDCBranchTy> MCDC;
  std::optional<DecisionTy> DecisionOpt;

  /// True or None.
  /// Stored in ColumnEnd:31.
  std::optional<bool> isGap;

  LocTy dLoc; ///< Differential line numbers.

  void mapping(llvm::yaml::IO &IO) override;

  void encode(raw_ostream &OS) const;
};

/// {NumRecs, Recs...}
struct FileRecsTy {
  std::vector<RecTy> Recs;

  void mapping(llvm::yaml::IO &IO);
};

/// An element of CovFun array.
struct CovFunTy {
  llvm::yaml::Hex64 NameRef;      ///< Hash value of the symbol.
  llvm::yaml::Hex64 FuncHash;     ///< Signature of this function.
  llvm::yaml::Hex64 FilenamesRef; ///< Pointer to CovMap
  std::vector<unsigned> FileIDs;  ///< Resolved by CovMap
  std::vector<ExpressionTy> Expressions;
  std::vector<FileRecsTy> Files; ///< 2-dimension array of Recs.

  void mapping(llvm::yaml::IO &IO);

  void encode(raw_ostream &OS, endianness Endianness) const;
};

/// An element of CovMap array.
struct CovMapTy {
  /// This is the key of CovMap but not present in the file
  /// format. Calculate and store with Filenames.
  llvm::yaml::Hex64 FilenamesRef;

  uint32_t Version;

  /// Raw Filenames (and storage of Files)
  std::vector<std::string> Filenames;

  void mapping(llvm::yaml::IO &IO);

  /// Encode Filenames. This is mostly used just to obtain FilenamesRef.
  std::pair<uint64_t, std::string> encodeFilenames(bool Compress = false) const;

  void encode(raw_ostream &OS, endianness Endianness) const;
};

} // namespace llvm::coverage::yaml

LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::coverage::yaml::CovMapTy)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::coverage::yaml::CovFunTy)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::coverage::yaml::ExpressionTy)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::coverage::yaml::RecTy)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::coverage::yaml::FileRecsTy)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(llvm::coverage::yaml::CounterTy)

#define LLVM_COVERAGE_YAML_ELEM_MAPPING(Ty)                                    \
  namespace llvm::yaml {                                                       \
  template <> struct MappingTraits<llvm::coverage::yaml::Ty> {                 \
    static void mapping(IO &IO, llvm::coverage::yaml::Ty &Obj) {               \
      Obj.mapping(IO);                                                         \
    }                                                                          \
  };                                                                           \
  }

/// `Flow` is used for emission of a compact oneliner for RecTy.
#define LLVM_COVERAGE_YAML_ELEM_MAPPING_FLOW(Ty)                               \
  namespace llvm::yaml {                                                       \
  template <> struct MappingTraits<llvm::coverage::yaml::Ty> {                 \
    static void mapping(IO &IO, llvm::coverage::yaml::Ty &Obj) {               \
      Obj.mapping(IO);                                                         \
      (void)flow;                                                              \
    }                                                                          \
    static const bool flow = true;                                             \
  };                                                                           \
  }

#define LLVM_COVERAGE_YAML_ENUM(Ty)                                            \
  namespace llvm::yaml {                                                       \
  template <> struct ScalarEnumerationTraits<llvm::coverage::yaml::Ty> {       \
    static void enumeration(IO &IO, llvm::coverage::yaml::Ty &Value);          \
  };                                                                           \
  }

LLVM_COVERAGE_YAML_ENUM(CounterTy::TagTy)
LLVM_COVERAGE_YAML_ENUM(RecTy::ExtTagTy)
LLVM_COVERAGE_YAML_ELEM_MAPPING_FLOW(CounterTy)
LLVM_COVERAGE_YAML_ELEM_MAPPING_FLOW(DecisionTy)
LLVM_COVERAGE_YAML_ELEM_MAPPING_FLOW(RecTy)
LLVM_COVERAGE_YAML_ELEM_MAPPING(FileRecsTy)
LLVM_COVERAGE_YAML_ELEM_MAPPING(CovFunTy)
LLVM_COVERAGE_YAML_ELEM_MAPPING(CovMapTy)

namespace llvm::covmap {

/// Returns whether Name is interested.
bool nameMatches(StringRef Name);

/// Returns a new ELFYAML Object.
std::unique_ptr<ELFYAML::CovMapSectionBase> make_unique(StringRef Name);

} // namespace llvm::covmap

#endif // LLVM_OBJECTYAML_COVMAP_H
