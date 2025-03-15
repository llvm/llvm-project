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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ObjectYAML/ELFYAML.h"
#include "llvm/ProfileData/Coverage/CoverageMappingWriter.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/LEB128.h"
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

void CounterTy::encode(raw_ostream &OS) const {
  encodeULEB128(Tag | (Val << 2), OS);
}

void DecisionTy::encode(raw_ostream &OS) const {
  encodeULEB128(BIdx, OS);
  encodeULEB128(NC, OS);
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
} // namespace

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
