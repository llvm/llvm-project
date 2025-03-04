//===- CovMap.cpp - ObjectYAML Interface for coverage map -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementations of CovMap and encoder.
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/CovMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ObjectYAML/ELFYAML.h"
#include "llvm/ProfileData/Coverage/CoverageMappingWriter.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#define COVMAP_V3

using namespace llvm;
using namespace llvm::coverage::yaml;
using namespace llvm::covmap;

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

void DecisionTy::encode(raw_ostream &OS) const {
  encodeULEB128(BIdx, OS);
  encodeULEB128(NC, OS);
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
  using PrfNamesTy = SmallVector<std::string>;
  SmallVector<PrfNamesTy, 1> PrfNames;

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
  DenseMap<uint64_t, struct CovMapTy *> CovMapByRef;

public:
  void collectCovMap(std::vector<CovMapTy> &CovMaps) {
    for (auto &CovMap : CovMaps)
      CovMapByRef[CovMap.FilenamesRef] = &CovMap;
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

std::unique_ptr<Encoder> Encoder::get(endianness Endianness) {
  return std::make_unique<EncoderImpl>(Endianness);
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

LLVM_YAML_IS_SEQUENCE_VECTOR(PrfNamesSection::PrfNamesTy)
