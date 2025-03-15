//===- CovMap.cpp - ObjectYAML Interface for coverage map -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementations of CovMap, encoder, decoder.
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/CovMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ObjectYAML/ELFYAML.h"
#include "llvm/ProfileData/Coverage/CoverageMappingReader.h"
#include "llvm/ProfileData/Coverage/CoverageMappingWriter.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#define COVMAP_V3

using namespace llvm;
using namespace llvm::coverage::yaml;
using namespace llvm::covmap;

bool Decoder::enabled;

Decoder::~Decoder() {}

// DataExtractor w/ single Cursor
struct coverage::yaml::DecoderContext : DataExtractor, DataExtractor::Cursor {
  DecoderContext(const ArrayRef<uint8_t> Content, bool IsLE)
      : DataExtractor(Content, IsLE, /*AddressSize=*/0),
        DataExtractor::Cursor(0) {}

  bool eof() { return DataExtractor::eof(*this); }
  uint32_t getU32() { return DataExtractor::getU32(*this); }
  uint64_t getU64() { return DataExtractor::getU64(*this); }
  Expected<uint64_t> getULEB128() {
    uint64_t Result = DataExtractor::getULEB128(*this);
    if (!*this)
      return takeError();
    return Result;
  }
  StringRef getBytes(size_t sz) { return DataExtractor::getBytes(*this, sz); }
};

void CounterTy::encode(raw_ostream &OS) const {
  encodeULEB128(Tag | (Val << 2), OS);
}

Error CounterTy::decodeOrTag(DecoderContext &Data) {
  auto COrErr = Data.getULEB128();
  if (!COrErr)
    return COrErr.takeError();
  Tag = static_cast<TagTy>(*COrErr & 0x03);
  Val = (*COrErr >> 2);
  return Error::success();
}

Error CounterTy::decode(DecoderContext &Data) {
  if (auto E = decodeOrTag(Data))
    return E;
  if (auto V = getExtTagVal())
    return make_error<CoverageMapError>(
        coveragemap_error::malformed,
        "Counter::Zero shouldn't have the Val: 0x" + Twine::utohexstr(V));
  return Error::success();
}

void DecisionTy::encode(raw_ostream &OS) const {
  encodeULEB128(BIdx, OS);
  encodeULEB128(NC, OS);
}

Error DecisionTy::decode(DecoderContext &Data) {
  auto BIdxOrErr = Data.getULEB128();
  if (!BIdxOrErr)
    return BIdxOrErr.takeError();
  BIdx = *BIdxOrErr;

  auto NCOrErr = Data.getULEB128();
  if (!NCOrErr)
    return NCOrErr.takeError();
  NC = *NCOrErr;

  return Error::success();
}

void RecTy::encode(raw_ostream &OS) const {
  if (Expansion) {
    encodeULEB128(4 + (*Expansion << 3), OS);
  } else if (ExtTag && *ExtTag == Skip) {
    encodeULEB128(2 << 3, OS);
  } else if (DecisionOpt) {
    assert(!ExtTag || *ExtTag == Decision);
    encodeULEB128(5 << 3, OS);
    DecisionOpt->encode(OS);
  } else if (MCDC) {
    assert(!ExtTag || *ExtTag == MCDCBranch);
    assert(BranchOpt);
    encodeULEB128(6 << 3, OS);
    (*BranchOpt)[0].encode(OS);
    (*BranchOpt)[1].encode(OS);
    encodeULEB128((*MCDC)[0], OS);
    encodeULEB128((*MCDC)[1], OS);
    encodeULEB128((*MCDC)[2], OS);
  } else if (BranchOpt) {
    assert(!ExtTag || *ExtTag == Branch);
    encodeULEB128(4 << 3, OS);
    (*BranchOpt)[0].encode(OS);
    (*BranchOpt)[1].encode(OS);
  } else {
    // Non-tag CounterTy
    CounterTy::encode(OS);
  }

  assert((!isGap || *isGap) && "Don't set isGap=false");
  uint32_t Gap = (isGap ? (1u << 31) : 0u);
  encodeULEB128(dLoc[0], OS);
  encodeULEB128(dLoc[1], OS);
  encodeULEB128(dLoc[2], OS);
  encodeULEB128(dLoc[3] | Gap, OS);
}

Error RecTy::decode(DecoderContext &Data) {
  auto getU16 = [&]() -> Expected<uint16_t> {
    auto ValOrErr = Data.getULEB128();
    if (!ValOrErr)
      return ValOrErr.takeError();
    if (*ValOrErr > 0x7FFF + 1)
      return make_error<CoverageMapError>(coveragemap_error::malformed,
                                          "MC/DC index is out of range: 0x" +
                                              Twine::utohexstr(*ValOrErr));
    return static_cast<uint16_t>(*ValOrErr);
  };

  auto decodeBranch = [&]() -> Error {
    auto &B = BranchOpt.emplace();
    if (auto E = B[0].decode(Data))
      return E;
    if (auto E = B[1].decode(Data))
      return E;
    return Error::success();
  };

  // Decode tagged CounterTy
  if (auto E = CounterTy::decodeOrTag(Data))
    return E;
  auto V = getExtTagVal();
  if (V == 0) {
    // Compatible to CounterTy
  } else if (V & 1u) {
    Expansion = (V >> 1);
  } else {
    auto Tag = (V >> 1);
    switch (Tag) {
    case Skip:
      ExtTag = Skip; // w/o Val
      break;
    case Decision:
      if (auto E = DecisionOpt.emplace().decode(Data))
        return E;
      ExtTag = Decision;
      break;
    case Branch:
      if (auto E = decodeBranch())
        return E;
      ExtTag = Branch;
      break;
    case MCDCBranch: {
      if (auto E = decodeBranch())
        return E;
      auto I0OrErr = getU16();
      if (!I0OrErr)
        return I0OrErr.takeError();
      auto I1OrErr = getU16();
      if (!I1OrErr)
        return I1OrErr.takeError();
      auto I2OrErr = getU16();
      if (!I2OrErr)
        return I2OrErr.takeError();
      MCDC = {*I0OrErr, *I1OrErr, *I2OrErr};
      ExtTag = MCDCBranch;
      break;
    }
    default:
      return make_error<CoverageMapError>(
          coveragemap_error::malformed,
          "Record doesn't have a valid Tag: 0x" + Twine::utohexstr(Tag));
    }
  }

  // Decode Loc
  auto LSDeltaOrErr = Data.getULEB128();
  if (!LSDeltaOrErr)
    return LSDeltaOrErr.takeError();

  auto CSOrErr = Data.getULEB128();
  if (!CSOrErr)
    return CSOrErr.takeError();

  auto NLOrErr = Data.getULEB128();
  if (!NLOrErr)
    return NLOrErr.takeError();

  auto CEOrErr = Data.getULEB128();
  if (!CEOrErr)
    return CEOrErr.takeError();
  auto ColumnEnd = *CEOrErr;

  // Gap is set in ColumnEnd:31
  if (ColumnEnd & (1u << 31))
    isGap = true;
  ColumnEnd &= ((1u << 31) - 1);

  dLoc = {*LSDeltaOrErr, *CSOrErr, *NLOrErr, ColumnEnd};

  return Error::success();
}

void CovFunTy::encode(raw_ostream &OS, endianness Endianness) const {
  // Encode Body in advance since DataSize should be known.
  std::string Body;
  raw_string_ostream SS(Body);

  encodeULEB128(FileIDs.size(), SS);
  for (auto I : FileIDs)
    encodeULEB128(I, SS);

  encodeULEB128(Expressions.size(), SS);
  for (const auto &[LHS, RHS] : Expressions) {
    LHS.encode(SS);
    RHS.encode(SS);
  }

  for (const auto &File : Files) {
    encodeULEB128(File.Recs.size(), SS);
    for (const auto &Rec : File.Recs)
      Rec.encode(SS);
  }

  // Emit the Header
  uint64_t NameRef = this->NameRef;
  uint32_t DataSize = Body.size();
  uint64_t FuncHash = this->FuncHash;
  char CoverageMapping = 0; // dummy
  uint64_t FilenamesRef = this->FilenamesRef;

#define COVMAP_FUNC_RECORD(Type, LLVMType, Name, Initializer)                  \
  if (sizeof(Name) > 1) {                                                      \
    Type t = support::endian::byte_swap(Name, Endianness);                     \
    OS << StringRef(reinterpret_cast<const char *>(&t), sizeof(t));            \
  }
#include "llvm/ProfileData/InstrProfData.inc"

  // Emit the body.
  OS << std::move(Body);
}

std::pair<uint64_t, std::string>
CovMapTy::encodeFilenames(bool Compress) const {
  std::string FilenamesBlob;
  llvm::raw_string_ostream OS(FilenamesBlob);
  CoverageFilenamesSectionWriter(this->Filenames).write(OS, Compress);

  return {llvm::IndexedInstrProf::ComputeHash(FilenamesBlob), FilenamesBlob};
}

Expected<uint64_t> CovFunTy::decode(const ArrayRef<uint8_t> Content,
                                    uint64_t Offset, endianness Endianness) {
  DecoderContext Data(Content, (Endianness == endianness::little));
  Data.seek(Offset);

  uint32_t DataSize;
  [[maybe_unused]] char CoverageMapping; // Ignored

#define COVMAP_FUNC_RECORD(Type, LLVMType, Name, Initializer)                  \
  if (sizeof(Type) == sizeof(uint64_t))                                        \
    Name = Data.getU64();                                                      \
  else if (sizeof(Type) == sizeof(uint32_t))                                   \
    Name = Data.getU32();                                                      \
  else                                                                         \
    assert(sizeof(Type) == sizeof(CoverageMapping) && "Unknown type");

#include "llvm/ProfileData/InstrProfData.inc"

  if (!Data)
    return Data.takeError();

  [[maybe_unused]] auto ExpectedEndOffset = Data.tell() + DataSize;

  // Decode body.
  auto NumFilesOrErr = Data.getULEB128();
  if (!NumFilesOrErr)
    return NumFilesOrErr.takeError();
  for (unsigned I = 0, E = *NumFilesOrErr; I != E; ++I) {
    if (auto IDOrErr = Data.getULEB128())
      FileIDs.push_back(*IDOrErr);
    else
      return IDOrErr.takeError();
  }

  auto NumExprOrErr = Data.getULEB128();
  if (!NumExprOrErr)
    return NumExprOrErr.takeError();
  Expressions.resize(*NumExprOrErr);
  for (auto &[LHS, RHS] : Expressions) {
    if (auto E = LHS.decode(Data))
      return std::move(E);
    if (auto E = RHS.decode(Data))
      return std::move(E);
  }

  for (unsigned FileIdx = 0; FileIdx != *NumFilesOrErr; ++FileIdx) {
    auto NumRegionsOrErr = Data.getULEB128();
    if (!NumRegionsOrErr)
      return NumRegionsOrErr.takeError();
    auto &File = Files.emplace_back();

    // Decode subarray.
    for (unsigned I = 0; I != *NumRegionsOrErr; ++I) {
      auto &Rec = File.Recs.emplace_back();
      if (auto E = Rec.decode(Data))
        return std::move(E);
    }
  }

  assert(Data.tell() == ExpectedEndOffset);
  return Data.tell();
}

void CovMapTy::encode(raw_ostream &OS, endianness Endianness) const {
  auto [FilenamesRef, FilenamesBlob] = encodeFilenames();

  uint32_t NRecords = 0;
  uint32_t FilenamesSize = FilenamesBlob.size();
  uint32_t CoverageSize = 0;
  uint32_t Version = this->Version;
  struct {
#define COVMAP_HEADER(Type, LLVMType, Name, Initializer) Type Name;
#include "llvm/ProfileData/InstrProfData.inc"
  } CovMapHeader = {
#define COVMAP_HEADER(Type, LLVMType, Name, Initializer)                       \
  support::endian::byte_swap(Name, Endianness),
#include "llvm/ProfileData/InstrProfData.inc"
  };
  StringRef HeaderBytes(reinterpret_cast<char *>(&CovMapHeader),
                        sizeof(CovMapHeader));
  OS << HeaderBytes;

  // llvm_covmap's alignment
  FilenamesBlob.resize(llvm::alignTo(FilenamesBlob.size(), sizeof(uint32_t)));
  OS << FilenamesBlob;
}

Expected<uint64_t> CovMapTy::decode(const ArrayRef<uint8_t> Content,
                                    uint64_t Offset, endianness Endianness) {
  DecoderContext Data(Content, (Endianness == endianness::little));
  Data.seek(Offset);

#define COVMAP_HEADER(Type, LLVMType, Name, Initializer)                       \
  static_assert(sizeof(Type) == sizeof(uint32_t));                             \
  Type Name = Data.getU32();
#include "llvm/ProfileData/InstrProfData.inc"
  if (!Data)
    return Data.takeError();
  assert(NRecords == 0);
  // +1: uint32_t FilenamesSize;
  assert(CoverageSize == 0);
  this->Version = Version;

  // Decode Body -- Filenames.
  StringRef FnBlob = Data.getBytes(FilenamesSize);
  if (!Data)
    return Data.takeError();
  this->FilenamesRef = MD5Hash(FnBlob);
  if (auto E = RawCoverageFilenamesReader(FnBlob, Filenames)
                   .read(static_cast<CovMapVersion>(Version)))
    return E;

  Offset = Data.tell();
  return Offset;
}

void CounterTy::mapping(llvm::yaml::IO &IO) {
  IO.mapRequired("Tag", Tag);
  IO.mapRequired("Val", Val);
}

void DecisionTy::mapping(llvm::yaml::IO &IO) {
  IO.mapRequired("BIdx", BIdx);
  IO.mapRequired("NCond", NC);
}

void RecTy::mapping(llvm::yaml::IO &IO) {
  IO.mapRequired("dLoc", dLoc);
  IO.mapOptional("isGap", isGap);
  CounterTy::mapping(IO);
  IO.mapOptional("ExtTag", ExtTag);
  IO.mapOptional("Expansion", Expansion);
  IO.mapOptional("Branch", BranchOpt);
  IO.mapOptional("MCDC", MCDC);
  IO.mapOptional("Decision", DecisionOpt);
}

void FileRecsTy::mapping(llvm::yaml::IO &IO) {
  IO.mapRequired("Regions", Recs);
}

void CovFunTy::mapping(llvm::yaml::IO &IO) {
  IO.mapRequired("NameRef", NameRef);
  IO.mapRequired("FuncHash", FuncHash);
  IO.mapRequired("FilenamesRef", FilenamesRef);
  IO.mapRequired("FileIDs", FileIDs);
  IO.mapRequired("Expressions", Expressions);
  IO.mapRequired("Files", Files);
}

void CovMapTy::mapping(llvm::yaml::IO &IO) {
  IO.mapRequired("FilenamesRef", FilenamesRef);
  IO.mapRequired("Version", Version);
  IO.mapRequired("Filenames", Filenames);
}

#define ECase(N, X) IO.enumCase(Value, #X, N::X)

void llvm::yaml::ScalarEnumerationTraits<CounterTy::TagTy>::enumeration(
    llvm::yaml::IO &IO, CounterTy::TagTy &Value) {
  ECase(CounterTy, Zero);
  ECase(CounterTy, Ref);
  ECase(CounterTy, Sub);
  ECase(CounterTy, Add);
}

void llvm::yaml::ScalarEnumerationTraits<RecTy::ExtTagTy>::enumeration(
    llvm::yaml::IO &IO, RecTy::ExtTagTy &Value) {
  ECase(RecTy, Skip);
  ECase(RecTy, Branch);
  ECase(RecTy, Decision);
  ECase(RecTy, MCDCBranch);
}

namespace {

struct PrfNamesSection : ELFYAML::CovMapSectionBase {
  InstrProfSymtab::PrfNamesChunksTy PrfNames;

  PrfNamesSection() { Name = "__llvm_prf_names"; }
  static bool nameMatches(StringRef Name) { return Name == "__llvm_prf_names"; }
  static bool classof(const Chunk *S) {
    return (isa<CovMapSectionBase>(S) && nameMatches(S->Name));
  }

  void mapping(llvm::yaml::IO &IO) override {
    IO.mapOptional("PrfNames", PrfNames);
  }

  Error encode(raw_ostream &OS, endianness Endianness) const override {
    for (const auto &Names : PrfNames) {
      std::string Result;
      if (auto E =
              collectGlobalObjectNameStrings(Names,
                                             /*doCompression=*/false, Result))
        return E;
      OS << Result;
    }
    return Error::success();
  }
};

struct CovMapSection : ELFYAML::CovMapSectionBase {
  std::vector<CovMapTy> CovMaps;

  CovMapSection() { Name = "__llvm_covmap"; }
  static bool nameMatches(StringRef Name) { return Name == "__llvm_covmap"; }
  static bool classof(const Chunk *S) {
    return (isa<CovMapSectionBase>(S) && nameMatches(S->Name));
  }

  void mapping(llvm::yaml::IO &IO) override {
    IO.mapOptional("CovMap", CovMaps);
  }

  Error decode(ArrayRef<uint8_t> Blob, unsigned AddressAlign,
               endianness Endianness) {
    uint64_t Offset = 0;

    while (true) {
      Offset = llvm::alignTo(Offset, AddressAlign);
      if (Offset >= Blob.size()) {
        break;
      }
      auto &CovMap = CovMaps.emplace_back();
      auto Result = CovMap.decode(Blob, Offset, Endianness);
      if (!Result) {
        return Result.takeError();
      }
      Offset = *Result;
    }

    return Error::success();
  }

  Error encode(raw_ostream &OS, endianness Endianness) const override {
    auto BaseOffset = OS.tell();
    for (const auto &CovMap : CovMaps) {
      OS.write_zeros(llvm::offsetToAlignment(OS.tell() - BaseOffset,
                                             llvm::Align(AddressAlign.value)));
      CovMap.encode(OS, Endianness);
    }
    return Error::success();
  }
};

struct CovFunSection : ELFYAML::CovMapSectionBase {
  std::vector<CovFunTy> CovFuns;

  CovFunSection() { Name = "__llvm_covfun"; }
  static bool nameMatches(StringRef Name) {
    return Name.starts_with("__llvm_covfun");
  }
  static bool classof(const Chunk *S) {
    return (isa<CovMapSectionBase>(S) && nameMatches(S->Name));
  }

  void mapping(llvm::yaml::IO &IO) override {
    IO.mapOptional("CovFun", CovFuns);
  }

  static Expected<std::vector<CovFunTy>> decode(ArrayRef<uint8_t> CovFunA,
                                                unsigned AddressAlign,
                                                endianness Endianness) {
    std::vector<CovFunTy> CovFuns;
    uint64_t Offset = 0;

    while (true) {
      Offset = llvm::alignTo(Offset, AddressAlign);
      if (Offset >= CovFunA.size())
        break;

      auto &CovFun = CovFuns.emplace_back();
      auto Result = CovFun.decode(CovFunA, Offset, Endianness);
      if (!Result)
        return Result.takeError();

      Offset = *Result;
    }

    return std::move(CovFuns);
  }

  Error encode(raw_ostream &OS, endianness Endianness) const override {
    auto BaseOffset = OS.tell();
    for (auto [I, CovFun] : enumerate(CovFuns)) {
      OS.write_zeros(llvm::offsetToAlignment(OS.tell() - BaseOffset,
                                             llvm::Align(AddressAlign.value)));
      CovFun.encode(OS, Endianness);
    }
    return Error::success();
  }
};

class DecoderImpl : public Decoder {
  std::unique_ptr<InstrProfSymtab> ProfileNames;
  std::vector<CovMapTy> TempCovMaps;

public:
  DecoderImpl(endianness Endianness, bool CovMapEnabled)
      : Decoder(Endianness), ProfileNames(std::make_unique<InstrProfSymtab>()) {
    enabled = CovMapEnabled;
  }

  Error acquire(unsigned AddressAlign, StringRef Name,
                ArrayRef<uint8_t> Content) override {
    // Don't register anything.
    if (!enabled)
      return Error::success();

    if (CovMapSection::nameMatches(Name)) {
      // Decode CovMaps in advance, since only CovMap knows its Version.
      // CovMaps is restored (into CovMapSection) later.
      auto TempCovMap = std::make_unique<CovMapSection>();
      if (auto E = TempCovMap->decode(Content, AddressAlign, Endianness))
        return E;
      TempCovMaps = std::move(TempCovMap->CovMaps);
    }

    return Error::success();
  }

  Error make(ELFYAML::CovMapSectionBase *Base,
             ArrayRef<uint8_t> Content) override {
    if (auto *S = dyn_cast<CovMapSection>(Base)) {
      // Store predecoded CovMaps.
      S->CovMaps = std::move(TempCovMaps);
      return Error::success();
    } else if (auto *S = dyn_cast<PrfNamesSection>(Base)) {
      // Decode PrfNames in advance since CovFun depends on it.
      auto PrfNamesOrErr = ProfileNames->createAndGetList(Content);
      if (!PrfNamesOrErr)
        return PrfNamesOrErr.takeError();
      S->PrfNames = std::move(*PrfNamesOrErr);
      return Error::success();
    } else if (auto *S = dyn_cast<CovFunSection>(Base)) {
      auto CovFunsOrErr =
          CovFunSection::decode(Content, S->AddressAlign, Endianness);
      if (!CovFunsOrErr)
        return CovFunsOrErr.takeError();
      S->CovFuns = std::move(*CovFunsOrErr);
      return Error::success();
    }

    llvm_unreachable("Unknown Section");
  }
};
} // namespace

std::unique_ptr<Decoder> Decoder::get(endianness Endianness,
                                      bool CovMapEnabled) {
  return std::make_unique<DecoderImpl>(Endianness, CovMapEnabled);
}

bool covmap::nameMatches(StringRef Name) {
  return (PrfNamesSection::nameMatches(Name) ||
          CovMapSection::nameMatches(Name) || CovFunSection::nameMatches(Name));
}

std::unique_ptr<ELFYAML::CovMapSectionBase>
covmap::make_unique(StringRef Name) {
  if (PrfNamesSection::nameMatches(Name))
    return std::make_unique<PrfNamesSection>();
  else if (CovMapSection::nameMatches(Name))
    return std::make_unique<CovMapSection>();
  else if (CovFunSection::nameMatches(Name))
    return std::make_unique<CovFunSection>();

  return nullptr;
}

LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::InstrProfSymtab::PrfNamesTy)
