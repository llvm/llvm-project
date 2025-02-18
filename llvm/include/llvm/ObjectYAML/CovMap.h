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
//   Provides YAML encoder and decoder for coverage map.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECTYAML_COVMAP_H
#define LLVM_OBJECTYAML_COVMAP_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ObjectYAML/ELFYAML.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/YAMLTraits.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <variant>

namespace llvm {
class InstrProfSymtab;
class raw_ostream;
} // namespace llvm

namespace llvm::coverage::yaml {

/// This works like vector container but can be replaced with
/// MutableArrayRef. See also SequenceTraits<VectorOrRef>.
template <typename T, typename Vec = std::vector<T>> class VectorOrRef {
  using Ref = MutableArrayRef<T>;

  /// Holds vector type initially.
  std::variant<Vec, Ref> Array;

public:
  // FIXME: Iterator impl is minimal easy.
  using iterator = T *;

  iterator begin() {
    if (auto *V = std::get_if<Vec>(&Array))
      return &V->front();
    return &std::get<Ref>(Array).front();
  }

  iterator end() {
    if (auto *V = std::get_if<Vec>(&Array))
      return &V->back() + 1;
    return &std::get<Ref>(Array).back() + 1;
  }

  size_t size() const {
    if (const auto *V = std::get_if<Vec>(&Array))
      return V->size();
    return std::get<Ref>(Array).size();
  }

  T &operator[](int Idx) {
    if (auto *V = std::get_if<Vec>(&Array))
      return (*V)[Idx];
    return std::get<Ref>(Array)[Idx];
  }

  void resize(size_t Size) { std::get<Vec>(Array).resize(Size); }

  VectorOrRef() = default;

  /// Initialize with MutableArrayRef.
  VectorOrRef(Ref &&Tmp) : Array(std::move(Tmp)) {}
};

/// Options for Decoder.
struct DecoderParam {
  bool Detailed; ///< Generate and show processed records.
  bool Raw;      ///< Show raw data oriented records.
  bool dLoc;     ///< Show raw dLoc (differential Loc).
};

struct DecoderContext;

/// Base Counter, corresponding to coverage::Counter.
struct CounterTy {
  enum TagTy : uint8_t {
    Zero = 0,
    Ref,
    Sub,
    Add,
  };

  /// Optional in detailed view, since most Tag can be determined from
  /// other optional fields.
  std::optional<TagTy> Tag;

  /// Internal use.
  std::optional<uint64_t> Val;

  std::optional<uint64_t> RefOpt;
  std::optional<uint64_t> SubOpt;
  std::optional<uint64_t> AddOpt;

  virtual ~CounterTy() {}

  virtual void mapping(llvm::yaml::IO &IO);

  /// Holds Val for extensions.
  Error decodeOrTag(DecoderContext &Data);

  /// Raise Error if Val isn't empty.
  Error decode(DecoderContext &Data);

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

  Error decode(DecoderContext &Data);

  void encode(raw_ostream &OS) const;
};

/// {LineStart, ColumnStart, LineEnd, ColumnEnd}
using LocTy = std::array<uint64_t, 4>;

///
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

  std::optional<LocTy> Loc;  ///< Absolute line numbers.
  std::optional<LocTy> dLoc; ///< Differential line numbers.

  void mapping(llvm::yaml::IO &IO) override;

  Error decode(DecoderContext &Data);

  void encode(uint64_t &StartLoc, raw_ostream &OS) const;
};

/// {NumRecs, Recs...}
struct FileRecsTy {
  std::optional<unsigned> Index;       ///< Shown in detailed view.
  std::optional<std::string> Filename; ///< Resolved by FileIDs.
  std::vector<RecTy> Recs;

  void mapping(llvm::yaml::IO &IO);
};

/// Key is FilenamesRef.
using CovMapByRefTy = llvm::DenseMap<uint64_t, struct CovMapTy *>;

/// An element of CovFun array.
struct CovFunTy {
  std::optional<llvm::yaml::Hex64> NameRef;     ///< Hash value of the symbol.
  std::optional<std::string> FuncName;          ///< Resolved by symtab.
  llvm::yaml::Hex64 FuncHash;                   ///< Signature of this function.
  llvm::yaml::Hex64 FilenamesRef;               ///< Pointer to CovMap
  std::optional<std::vector<unsigned>> FileIDs; ///< Resolved by CovMap
  std::vector<ExpressionTy> Expressions;
  std::vector<FileRecsTy> Files; ///< 2-dimension array of Recs.

  void mapping(llvm::yaml::IO &IO);

  /// Depends on CovMap and SymTab(IPSK_names)
  Expected<uint64_t> decode(CovMapByRefTy &CovMapByRef, InstrProfSymtab *SymTab,
                            const ArrayRef<uint8_t> Content, uint64_t Offset,
                            endianness Endianness, const DecoderParam &Param);

  void encode(raw_ostream &OS, endianness Endianness) const;
};

/// An element of CovMap array.
struct CovMapTy {
  /// This is the key of CovMap but not present in the file
  /// format. Calculate and store with Filenames.
  llvm::yaml::Hex64 FilenamesRef;

  std::optional<uint32_t> Version;

  /// Raw Filenames (and storage of Files)
  std::optional<std::vector<std::string>> Filenames;

  /// Since Version5: Filenames[0] is the working directory (or
  /// zero-length string). Note that indices in CovFun::FileIDs is
  /// base on Filenames. (Then, +0, as WD, is not expected to appear)
  std::optional<std::string> WD;
  /// This may be ArrayRef in Decoder since Filenames has been
  /// filled. On the other hand in Encoder, this should be a vector
  /// since YAML parser doesn't endorse references.
  std::optional<VectorOrRef<std::string>> Files;

  void mapping(llvm::yaml::IO &IO);

  bool useWD() const { return (!Version || *Version >= 4); }
  StringRef getWD() const { return (WD ? *WD : StringRef()); }

  Expected<uint64_t> decode(const ArrayRef<uint8_t> Content, uint64_t Offset,
                            endianness Endianness, const DecoderParam &Param);

  /// Generate Accumulated list with WD.
  /// Returns a single element {WD} if AccFiles is not given.
  std::vector<std::string>
  generateAccFilenames(const std::optional<ArrayRef<StringRef>> &AccFilesOpt =
                           std::nullopt) const;
  /// Regenerate Filenames with WD.
  /// Use Files if it is not None. Or given AccFiles is used.
  void
  regenerateFilenames(const std::optional<ArrayRef<StringRef>> &AccFilesOpt);

  /// Encode Filenames. This is mostly used just to obtain FilenamesRef.
  std::pair<uint64_t, std::string> encodeFilenames(
      const std::optional<ArrayRef<StringRef>> &AccFilesOpt = std::nullopt,
      bool Compress = false) const;

  void encode(raw_ostream &OS, endianness Endianness) const;
};

} // namespace llvm::coverage::yaml

namespace llvm::yaml {
template <typename T>
struct SequenceTraits<llvm::coverage::yaml::VectorOrRef<T>> {
  static size_t size(IO &io, llvm::coverage::yaml::VectorOrRef<T> &seq) {
    return seq.size();
  }
  static T &element(IO &, llvm::coverage::yaml::VectorOrRef<T> &seq,
                    size_t index) {
    if (index >= seq.size())
      seq.resize(index + 1);
    return seq[index];
  }
};
} // namespace llvm::yaml

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

class Decoder {
protected:
  endianness Endianness;

public:
  Decoder(endianness Endianness) : Endianness(Endianness) {}
  virtual ~Decoder() {}

  /// Returns DecoderImpl.
  static std::unique_ptr<Decoder>
  get(endianness Endianness, const coverage::yaml::DecoderParam &Param);

  /// Called from the Sections loop in advance of the final dump.
  /// Decoder predecodes Names and CovMap, and captures Contents of
  /// CovFuns.
  virtual Error
  acquire(uint64_t Offset, unsigned AddressAlign, StringRef Name,
          std::function<Expected<ArrayRef<uint8_t>>()> getSectionContents) = 0;

  /// Called before the final dump after `acquire`.
  /// Decode contents partially and resolve names.
  virtual Error fixup() = 0;

  /// Create an ELFYAML object. This just puts predecoded data in
  /// `fixup`.
  virtual Expected<ELFYAML::Section *>
  make(uint64_t Offset, StringRef Name,
       std::function<Error(ELFYAML::Section &S)> dumpCommonSection) = 0;

  /// Suppress emission of CovMap unless enabled.
  static bool enabled;
};

class Encoder {
protected:
  endianness Endianness;

public:
  Encoder(endianness Endianness) : Endianness(Endianness) {}
  virtual ~Encoder() {}

  /// Returns EncoderImpl.
  static std::unique_ptr<Encoder> get(endianness Endianness);

  /// Called from the Sections loop.
  virtual void collect(ELFYAML::Chunk *Chunk) = 0;

  /// Resolves names along DecoderParam in advance of Emitter. It'd be
  /// too late to resolve sections in Emitter since they are immutable
  /// then.
  virtual void fixup() = 0;
};

/// Returns whether Name is interested.
bool nameMatches(StringRef Name);

/// Returns a new ELFYAML Object.
std::unique_ptr<ELFYAML::Section> make_unique(StringRef Name);

} // namespace llvm::covmap

#endif // LLVM_OBJECTYAML_COVMAP_H
