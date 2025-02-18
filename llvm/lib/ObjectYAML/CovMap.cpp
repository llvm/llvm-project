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
#include "llvm/ADT/MapVector.h"
#include "llvm/ObjectYAML/ELFYAML.h"
#include "llvm/ProfileData/Coverage/CoverageMapping.h"
#include "llvm/ProfileData/Coverage/CoverageMappingReader.h"
#include "llvm/ProfileData/Coverage/CoverageMappingWriter.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/YAMLTraits.h"
#include <cstdint>

#define COVMAP_V3

using namespace llvm;
using namespace llvm::coverage::yaml;
using namespace llvm::covmap;

bool Decoder::enabled;

// DataExtractor w/ single Cursor
struct coverage::yaml::DecoderContext : DataExtractor,
                                        DataExtractor::Cursor,
                                        DecoderParam {
  uint64_t LineStart = 0;

  DecoderContext(const ArrayRef<uint8_t> Content, const DecoderParam &Param,
                 bool IsLE)
      : DataExtractor(Content, IsLE, /*AddressSize=*/0),
        DataExtractor::Cursor(0), DecoderParam(Param) {}

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
  std::pair<unsigned, uint64_t> C;
  if (RefOpt)
    C = {Ref, *RefOpt};
  else if (SubOpt)
    C = {Sub, *SubOpt};
  else if (AddOpt)
    C = {Add, *AddOpt};
  else if (Tag && *Tag == Zero)
    C = {Zero, 0u};
  else if (Tag && Val)
    C = {*Tag, *Val};
  else
    llvm_unreachable("Null value cannot be met");

  encodeULEB128(C.first | (C.second << 2), OS);
}

Error CounterTy::decodeOrTag(DecoderContext &Data) {
  auto COrErr = Data.getULEB128();
  if (!COrErr)
    return COrErr.takeError();
  auto T = static_cast<TagTy>(*COrErr & 0x03);
  auto V = (*COrErr >> 2);
  if (T == Zero) {
    if (V == 0)
      Tag = Zero; // w/o Val
    else
      Val = V; // w/o Tag
  } else {
    if (Data.Raw) {
      Tag = T;
      Val = V;
    } else {
      switch (T) {
      case Zero:
        llvm_unreachable("Zero should be handled in advance");
      case Ref:
        RefOpt = V;
        break;
      case Sub:
        SubOpt = V;
        break;
      case Add:
        AddOpt = V;
        break;
      }
    }
  }

  return Error::success();
}

Error CounterTy::decode(DecoderContext &Data) {
  if (auto E = decodeOrTag(Data))
    return E;
  if (!this->Tag && this->Val)
    return make_error<CoverageMapError>(
        coveragemap_error::malformed,
        "Counter::Zero shouldn't have the Val: 0x" +
            Twine::utohexstr(*this->Val));
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

void RecTy::encode(uint64_t &StartLoc, raw_ostream &OS) const {
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
  if (Loc) {
    encodeULEB128((*Loc)[0] - StartLoc, OS);
    encodeULEB128((*Loc)[1], OS);
    encodeULEB128((*Loc)[2] - (*Loc)[0], OS);
    encodeULEB128((*Loc)[3] | Gap, OS);
    StartLoc = (*Loc)[0];
  } else {
    encodeULEB128((*dLoc)[0], OS);
    encodeULEB128((*dLoc)[1], OS);
    encodeULEB128((*dLoc)[2], OS);
    encodeULEB128((*dLoc)[3] | Gap, OS);
  }
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
  if (!this->Val || this->Tag) {
    // Compatible to CounterTy
  } else if (*this->Val & 1u) {
    Expansion = (*this->Val >> 1);
    this->Val.reset();
  } else {
    auto Tag = *this->Val >> 1;
    this->Val.reset();
    switch (Tag) {
    case Skip:
      ExtTag = Skip; // w/o Val
      break;
    case Decision:
      if (auto E = DecisionOpt.emplace().decode(Data))
        return E;
      if (Data.Raw)
        ExtTag = Decision;
      break;
    case Branch:
      if (auto E = decodeBranch())
        return E;
      if (Data.Raw)
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
      if (Data.Raw)
        ExtTag = MCDCBranch;
      break;
    }
    default:
      return make_error<CoverageMapError>(
          coveragemap_error::malformed,
          "Record doesn't have an valid Tag: 0x" + Twine::utohexstr(Tag));
    }
  }

  // Decode Loc
  auto LSDeltaOrErr = Data.getULEB128();
  if (!LSDeltaOrErr)
    return LSDeltaOrErr.takeError();
  Data.LineStart += *LSDeltaOrErr;

  auto CSOrErr = Data.getULEB128();
  if (!CSOrErr)
    return CSOrErr.takeError();

  auto NLOrErr = Data.getULEB128();
  if (!NLOrErr)
    return NLOrErr.takeError();
  auto LineEnd = Data.LineStart + *NLOrErr;

  auto CEOrErr = Data.getULEB128();
  if (!CEOrErr)
    return CEOrErr.takeError();
  auto ColumnEnd = *CEOrErr;

  // Gap is set in ColumnEnd:31
  if (ColumnEnd & (1u << 31))
    isGap = true;
  ColumnEnd &= ((1u << 31) - 1);

  dLoc = {*LSDeltaOrErr, *CSOrErr, *NLOrErr, ColumnEnd};
  Loc = {Data.LineStart, *CSOrErr, LineEnd, ColumnEnd};

  return Error::success();
}

void CovFunTy::encode(raw_ostream &OS, endianness Endianness) const {
  // Encode Body in advance since DataSize should be known.
  std::string Body;
  raw_string_ostream SS(Body);

  assert(FileIDs);
  encodeULEB128(FileIDs->size(), SS);
  for (auto I : *FileIDs)
    encodeULEB128(I, SS);

  encodeULEB128(Expressions.size(), SS);
  for (const auto &[LHS, RHS] : Expressions) {
    LHS.encode(SS);
    RHS.encode(SS);
  }

  for (const auto &File : Files) {
    encodeULEB128(File.Recs.size(), SS);
    uint64_t StartLoc = 0;
    for (const auto &Rec : File.Recs)
      Rec.encode(StartLoc, SS);
  }

  // Emit the Header
  uint64_t NameRef = (this->NameRef ? static_cast<uint64_t>(*this->NameRef)
                                    : MD5Hash(*this->FuncName));
  uint32_t DataSize = Body.size();
  /* this->FuncHash */
  char CoverageMapping = 0; // dummy
  /* this->FilenamesRef */

#define COVMAP_FUNC_RECORD(Type, LLVMType, Name, Initializer)                  \
  if (sizeof(Name) > 1) {                                                      \
    Type t = support::endian::byte_swap(Name, Endianness);                     \
    OS << StringRef(reinterpret_cast<const char *>(&t), sizeof(t));            \
  }
#include "llvm/ProfileData/InstrProfData.inc"

  // Emit the body.
  OS << std::move(Body);
}

std::vector<std::string> CovMapTy::generateAccFilenames(
    const std::optional<ArrayRef<StringRef>> &AccFilesOpt) const {
  std::vector<std::string> Result;
  if (useWD())
    Result.push_back(getWD().str());
  // Returns {WD} if AccFiles is None.
  if (AccFilesOpt) {
    for (auto &Filename : *AccFilesOpt)
      Result.push_back(Filename.str());
  }
  return Result;
}

void CovMapTy::regenerateFilenames(
    const std::optional<ArrayRef<StringRef>> &AccFilesOpt) {
  assert(!this->Filenames);
  if (this->Files) {
    auto &CovMapFilenames = this->Filenames.emplace(generateAccFilenames());
    assert(CovMapFilenames.size() <= 1);
    for (auto &&File : *this->Files)
      CovMapFilenames.push_back(std::move(File));
  } else {
    // Encode Accfiles, that comes from CovFun.
    this->Filenames = generateAccFilenames(AccFilesOpt);
  }
}

std::pair<uint64_t, std::string>
CovMapTy::encodeFilenames(const std::optional<ArrayRef<StringRef>> &AccFilesOpt,
                          bool Compress) const {
  ArrayRef<std::string> TempFilenames;
  std::vector<std::string> AccFilenames; // Storage

  if (AccFilesOpt) {
    AccFilenames = generateAccFilenames(AccFilesOpt);
    TempFilenames = AccFilenames;
  } else {
    assert(this->Filenames);
    TempFilenames = ArrayRef(*this->Filenames);
  }

  std::string FilenamesBlob;
  llvm::raw_string_ostream OS(FilenamesBlob);
  CoverageFilenamesSectionWriter(TempFilenames).write(OS, Compress);

  return {llvm::IndexedInstrProf::ComputeHash(FilenamesBlob), FilenamesBlob};
}

Expected<uint64_t> CovFunTy::decode(CovMapByRefTy &CovMapByRef,
                                    InstrProfSymtab *Symtab,
                                    const ArrayRef<uint8_t> Content,
                                    uint64_t Offset, endianness Endianness,
                                    const DecoderParam &Param) {
  DecoderContext Data(Content, Param, (Endianness == endianness::little));
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

  if (Data.Detailed)
    FuncName = Symtab->getFuncOrVarNameIfDefined(*NameRef);

  if (!Data.Raw)
    NameRef.reset();

  [[maybe_unused]] auto ExpectedEndOffset = Data.tell() + DataSize;

  // Decode body.
  assert(CovMapByRef.contains(this->FilenamesRef));
  auto &CovMap = *CovMapByRef[this->FilenamesRef];
  FileIDs.emplace();

  auto NumFilesOrErr = Data.getULEB128();
  if (!NumFilesOrErr)
    return NumFilesOrErr.takeError();
  for (unsigned I = 0, E = *NumFilesOrErr; I != E; ++I) {
    if (auto IDOrErr = Data.getULEB128())
      FileIDs->push_back(*IDOrErr);
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
    if (Data.Detailed) {
      File.Index = FileIdx; // Sequential number.
      File.Filename = (*CovMap.Filenames)[(*FileIDs)[FileIdx]];
    }

    // Decode subarray.
    Data.LineStart = 0;
    for (unsigned I = 0; I != *NumRegionsOrErr; ++I) {
      auto &Rec = File.Recs.emplace_back();
      if (auto E = Rec.decode(Data))
        return std::move(E);

      // Hide either Loc or dLoc.
      if (!Data.Detailed || Data.dLoc)
        Rec.Loc.reset();
      else if (!Data.Raw)
        Rec.dLoc.reset();
    }
  }

  // Hide FileIDs.
  if (!Data.Raw)
    FileIDs.reset();

  assert(Data.tell() == ExpectedEndOffset);
  return Data.tell();
}

void CovMapTy::encode(raw_ostream &OS, endianness Endianness) const {
  auto [FilenamesRef, FilenamesBlob] = encodeFilenames();

  uint32_t NRecords = 0;
  uint32_t FilenamesSize = FilenamesBlob.size();
  uint32_t CoverageSize = 0;
  uint32_t Version =
      (this->Version ? *this->Version : INSTR_PROF_COVMAP_VERSION);
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
                                    uint64_t Offset, endianness Endianness,
                                    const DecoderParam &Param) {
  DecoderContext Data(Content, Param, (Endianness == endianness::little));
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
  this->Filenames.emplace();
  if (auto E = RawCoverageFilenamesReader(FnBlob, *this->Filenames)
                   .read(static_cast<CovMapVersion>(Version)))
    return E;

  if (Param.Detailed && useWD()) {
    assert(this->Filenames->size() >= 1);
    auto FilenamesI = this->Filenames->begin();
    StringRef WD = *FilenamesI++;
    if (!WD.empty())
      this->WD = WD;
    // Use Filenames as a storage.
    this->Files.emplace(
        MutableArrayRef(&*FilenamesI, &*this->Filenames->end()));
  }

  Offset = Data.tell();
  return Offset;
}

void CounterTy::mapping(llvm::yaml::IO &IO) {
  IO.mapOptional("Tag", Tag);
  IO.mapOptional("Val", Val);
  IO.mapOptional("Ref", RefOpt);
  IO.mapOptional("Sub", SubOpt);
  IO.mapOptional("Add", AddOpt);
}

void DecisionTy::mapping(llvm::yaml::IO &IO) {
  IO.mapRequired("BIdx", BIdx);
  IO.mapRequired("NCond", NC);
}

void RecTy::mapping(llvm::yaml::IO &IO) {
  IO.mapOptional("Loc", Loc);
  IO.mapOptional("dLoc", dLoc);
  IO.mapOptional("isGap", isGap);
  CounterTy::mapping(IO);
  IO.mapOptional("ExtTag", ExtTag);
  IO.mapOptional("Expansion", Expansion);
  IO.mapOptional("Branch", BranchOpt);
  IO.mapOptional("MCDC", MCDC);
  IO.mapOptional("Decision", DecisionOpt);
}

void FileRecsTy::mapping(llvm::yaml::IO &IO) {
  IO.mapOptional("Index", Index);
  IO.mapOptional("Filename", Filename);
  IO.mapRequired("Regions", Recs);
}

void CovFunTy::mapping(llvm::yaml::IO &IO) {
  IO.mapOptional("NameRef", NameRef);
  IO.mapOptional("FuncName", FuncName);
  IO.mapRequired("FuncHash", FuncHash);
  IO.mapRequired("FilenamesRef", FilenamesRef);
  IO.mapOptional("FileIDs", FileIDs);
  IO.mapRequired("Expressions", Expressions);
  IO.mapRequired("Files", Files);
}

void CovMapTy::mapping(llvm::yaml::IO &IO) {
  IO.mapRequired("FilenamesRef", FilenamesRef);
  IO.mapOptional("Version", Version);

  if (!WD && !Files)
    // Suppress this regardless of (Detailed && Raw).
    // Since it is obviously redundant.
    IO.mapOptional("Filenames", Filenames);

  IO.mapOptional("WD", WD);
  IO.mapOptional("Files", Files);
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
               endianness Endianness, const DecoderParam &Param) {
    uint64_t Offset = 0;

    while (true) {
      Offset = llvm::alignTo(Offset, AddressAlign);
      if (Offset >= Blob.size()) {
        break;
      }
      auto &CovMap = CovMaps.emplace_back();
      auto Result = CovMap.decode(Blob, Offset, Endianness, Param);
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

  static Expected<std::vector<CovFunTy>>
  decode(CovMapByRefTy &CovMapByRef, InstrProfSymtab *Symtab,
         ArrayRef<uint8_t> CovFunA, unsigned AddressAlign,
         endianness Endianness, const DecoderParam &Param) {
    std::vector<CovFunTy> CovFuns;
    uint64_t Offset = 0;

    while (true) {
      Offset = llvm::alignTo(Offset, AddressAlign);
      if (Offset >= CovFunA.size())
        break;

      auto &CovFun = CovFuns.emplace_back();
      auto Result = CovFun.decode(CovMapByRef, Symtab, CovFunA, Offset,
                                  Endianness, Param);
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

class CovMapFilenamesResolver {
  DenseMap<uint64_t, SetVector<StringRef>> FilenamesByCovMap;
  std::vector<CovFunTy *> UnresolvedCovFuns;

protected:
  CovMapByRefTy CovMapByRef;
  std::vector<CovMapTy> TempCovMaps; // For Decoder

public:
  void collectCovMap(std::vector<CovMapTy> &CovMaps) {
    for (auto &CovMap : CovMaps)
      CovMapByRef[CovMap.FilenamesRef] = &CovMap;
  }

  void moveAndCollectCovMap(std::vector<CovMapTy> &&CovMaps) {
    TempCovMaps = std::move(CovMaps);
    collectCovMap(TempCovMaps);
  }

  void collectCovFunFilenames(std::vector<CovFunTy> &CovFuns) {
    for (auto &CovFun : CovFuns) {
      auto &Filenames = FilenamesByCovMap[CovFun.FilenamesRef];
      for (const auto &File : CovFun.Files) {
        if (!File.Filename)
          goto skip;
        Filenames.insert(*File.Filename);
      }
      UnresolvedCovFuns.push_back(&CovFun);
    skip:;
    }
  }

  void decMayeResetFilenames(std::vector<CovMapTy> &CovMaps) {
    for (auto &CovMap : CovMaps) {
      auto FilenamesI = FilenamesByCovMap.find(CovMap.FilenamesRef);
      if (FilenamesI == FilenamesByCovMap.end())
        continue;

      // Calculate FilenamesRef with Filenames from CovFuns.
      // If matches, hide Filenames from CovMap.
      auto [AccFilenamesRef, _] =
          CovMap.encodeFilenames(FilenamesI->second.getArrayRef());
      if (CovMap.FilenamesRef == AccFilenamesRef) {
        CovMap.Files.reset();
        CovMap.Filenames.reset(); // FilenamesI has been invalidated.
      }
    }
  }

  void encFixup() {
    for (auto &[_, CovMap] : CovMapByRef) {
      auto FilenamesI = FilenamesByCovMap.find(CovMap->FilenamesRef);
      if (FilenamesI != FilenamesByCovMap.end()) {
        // Check Filenames satisfies covfuns
        DenseSet<StringRef> FilenamesSet;
        if (CovMap->Files) {
          for (const auto &Filename : *CovMap->Files)
            FilenamesSet.insert(Filename);
        } else if (CovMap->Filenames) {
          for (const auto &Filename : *CovMap->Filenames)
            FilenamesSet.insert(Filename);
        }

        for (const auto &Filename : FilenamesI->second) {
          if (!FilenamesSet.contains(Filename)) {
            // If not, regenerate Filenames.
            CovMap->Files.reset();
            CovMap->Filenames.reset();
            break;
          }
        }
      }

      if (!CovMap->Filenames) {
        // Regenerate.
        // Use Files if exists.
        // Use CovFuns (FilenamesI) otherwise.
        assert(CovMap->Files || FilenamesI != FilenamesByCovMap.end());
        CovMap->regenerateFilenames(
            CovMap->Files ? std::nullopt : FilenamesI->second.getArrayRef());
      }
      auto [FilenamesRef, FilenamesBlob] = CovMap->encodeFilenames();
      assert(CovMap->FilenamesRef == FilenamesRef);
    }

    // Fill FileIDs
    for (auto *CovFun : UnresolvedCovFuns) {
      assert(CovMapByRef[CovFun->FilenamesRef]);
      assert(CovMapByRef[CovFun->FilenamesRef]->Filenames);
      const auto &CovMapFilenames =
          *CovMapByRef[CovFun->FilenamesRef]->Filenames;
      auto &FileIDs = CovFun->FileIDs.emplace();
      for (const auto &File : CovFun->Files) {
        auto I = std::find(CovMapFilenames.begin(), CovMapFilenames.end(),
                           File.Filename);
        assert(I != CovMapFilenames.end());
        FileIDs.push_back(std::distance(CovMapFilenames.begin(), I));
      }
      assert(CovFun->Files.size() == FileIDs.size());
    }
  }
};

class DecoderImpl : public Decoder, CovMapFilenamesResolver {
  DecoderParam Param;

  std::unique_ptr<InstrProfSymtab> ProfileNames;

  InstrProfSymtab::PrfNamesChunksTy PrfNames;

  MapVector<uint64_t, std::pair<ArrayRef<uint8_t>, unsigned>> CovFunBlobs;
  DenseMap<uint64_t, std::vector<CovFunTy>> TempCovFuns;

public:
  DecoderImpl(endianness Endianness, const DecoderParam &Param)
      : Decoder(Endianness), Param(Param),
        ProfileNames(std::make_unique<InstrProfSymtab>()) {
    enabled = (Param.Detailed || Param.Raw);
  }

  Error acquire(uint64_t Offset, unsigned AddressAlign, StringRef Name,
                std::function<Expected<ArrayRef<uint8_t>>()> getSectionContents)
      override {
    // Don't register anything.
    if (!enabled)
      return Error::success();

    if (PrfNamesSection::nameMatches(Name)) {
      auto ContentOrErr = getSectionContents();
      if (!ContentOrErr)
        return ContentOrErr.takeError();
      // Decode PrfNames in advance since CovFun depends on it.
      auto PrfNamesOrErr = ProfileNames->createAndGetList(*ContentOrErr);
      if (!PrfNamesOrErr)
        return PrfNamesOrErr.takeError();
      PrfNames = std::move(*PrfNamesOrErr);
    } else if (CovMapSection::nameMatches(Name)) {
      auto ContentOrErr = getSectionContents();
      if (!ContentOrErr)
        return ContentOrErr.takeError();

      // Decode CovMaps in advance, since only CovMap knows its Version.
      // CovMaps is restored (into CovMapSection) later.
      auto TempCovMap = std::make_unique<CovMapSection>();
      if (auto E = TempCovMap->decode(*ContentOrErr, AddressAlign, Endianness,
                                      Param))
        return E;
      moveAndCollectCovMap(std::move(TempCovMap->CovMaps));
    } else if (CovFunSection::nameMatches(Name)) {
      auto ContentOrErr = getSectionContents();
      if (!ContentOrErr)
        return ContentOrErr.takeError();

      // Will be decoded after CovMap is met.
      CovFunBlobs[Offset] = {*ContentOrErr, AddressAlign};
    }

    return Error::success();
  }

  Error fixup() override {
    // Decode CovFun(s) with predecoded PrfNames and CovMap.
    for (const auto &[Offset, CovFunBlob] : CovFunBlobs) {
      auto CovFunsOrErr = CovFunSection::decode(
          CovMapByRef, ProfileNames.get(), CovFunBlob.first, CovFunBlob.second,
          Endianness, Param);
      if (!CovFunsOrErr)
        return CovFunsOrErr.takeError();
      TempCovFuns[Offset] = std::move(*CovFunsOrErr);
      collectCovFunFilenames(TempCovFuns[Offset]);
    }
    return Error::success();
  }

  Expected<ELFYAML::Section *>
  make(uint64_t Offset, StringRef Name,
       std::function<Error(ELFYAML::Section &S)> dumpCommonSection) override {
    if (PrfNamesSection::nameMatches(Name)) {
      auto S = std::make_unique<PrfNamesSection>();
      if (Error E = dumpCommonSection(*S))
        return std::move(E);
      S->PrfNames = std::move(PrfNames);
      return S.release();
    } else if (CovMapSection::nameMatches(Name)) {
      auto S = std::make_unique<CovMapSection>();
      if (Error E = dumpCommonSection(*S))
        return std::move(E);

      // Store predecoded CovMaps.
      S->CovMaps = std::move(TempCovMaps);

      // Hide Filenames if it is reproducible from CovFuns.
      if (Param.Detailed)
        decMayeResetFilenames(S->CovMaps);

      return S.release();
    } else if (CovFunSection::nameMatches(Name)) {
      auto S = std::make_unique<CovFunSection>();
      if (Error E = dumpCommonSection(*S))
        return std::move(E);

      assert(S->CovFuns.empty());
      assert(TempCovFuns.contains(Offset));
      S->CovFuns = std::move(TempCovFuns[Offset]);

      return S.release();
    }

    llvm_unreachable("Name didn't match");
  }
};

class EncoderImpl : public Encoder, CovMapFilenamesResolver {
public:
  EncoderImpl(endianness Endianness) : Encoder(Endianness) {}

  void collect(ELFYAML::Chunk *Chunk) override {
    if (auto S = dyn_cast<CovMapSection>(Chunk)) {
      collectCovMap(S->CovMaps);
    } else if (auto S = dyn_cast<CovFunSection>(Chunk)) {
      collectCovFunFilenames(S->CovFuns);
    }
  }

  void fixup() override { encFixup(); }
};
} // namespace

std::unique_ptr<Decoder> Decoder::get(endianness Endianness,
                                      const DecoderParam &Param) {
  return std::make_unique<DecoderImpl>(Endianness, Param);
}

std::unique_ptr<Encoder> Encoder::get(endianness Endianness) {
  return std::make_unique<EncoderImpl>(Endianness);
}

bool covmap::nameMatches(StringRef Name) {
  return (PrfNamesSection::nameMatches(Name) ||
          CovMapSection::nameMatches(Name) || CovFunSection::nameMatches(Name));
}

std::unique_ptr<ELFYAML::Section> covmap::make_unique(StringRef Name) {
  if (PrfNamesSection::nameMatches(Name))
    return std::make_unique<PrfNamesSection>();
  else if (CovMapSection::nameMatches(Name))
    return std::make_unique<CovMapSection>();
  else if (CovFunSection::nameMatches(Name))
    return std::make_unique<CovFunSection>();

  return nullptr;
}

LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::InstrProfSymtab::PrfNamesTy)
