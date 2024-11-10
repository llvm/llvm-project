//===- BinaryObjectFile.cpp - Binary object file implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of the BinaryObjectFile class implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/BinaryObjectFile.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/Triple.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

using namespace llvm;
using namespace object;

BinaryObjectFile::BinaryObjectFile(MemoryBufferRef Source)
    : ObjectFile(ID_Binary, Source), Symbols(), Sections() {
  Symbols.push_back(BinarySymbol{});
  Symbols[0].Name = ".data";
  Sections.push_back(BinarySection{});
  Sections[0].Size = Source.getBufferSize();
}

Expected<std::unique_ptr<BinaryObjectFile>>
ObjectFile::createBinaryObjectFile(MemoryBufferRef Obj) {
  Error Err = Error::success();
  auto ObjectFile = std::make_unique<BinaryObjectFile>(Obj);
  return std::move(ObjectFile);
}

bool BinaryObjectFile::is64Bit() const { return true; }

basic_symbol_iterator BinaryObjectFile::symbol_begin() const {
  DataRefImpl Ref;
  Ref.d.a = 1; // Arbitrary non-zero value so that Ref.p is non-null
  Ref.d.b = 0; // Symbol index
  return BasicSymbolRef(Ref, this);
}

basic_symbol_iterator BinaryObjectFile::symbol_end() const {
  DataRefImpl Ref;
  Ref.d.a = 1; // Arbitrary non-zero value so that Ref.p is non-null
  Ref.d.b = Symbols.size(); // Symbol index
  return BasicSymbolRef(Ref, this);
}

void BinaryObjectFile::moveSymbolNext(DataRefImpl &Symb) const { Symb.d.b++; }

const BinarySymbol &
BinaryObjectFile::getBinarySymbol(const DataRefImpl &Symb) const {
  assert(Symb.d.b < Symbols.size());
  return Symbols[Symb.d.b];
}

const BinarySymbol &
BinaryObjectFile::getBinarySymbol(const SymbolRef &Symb) const {
  return getBinarySymbol(Symb.getRawDataRefImpl());
}

const BinarySection &
BinaryObjectFile::getBinarySection(const DataRefImpl Ref) const {
  assert(Ref.d.a < Sections.size());
  return Sections[Ref.d.a];
}

const BinarySection &
BinaryObjectFile::getBinarySection(const SectionRef &Section) const {
  return getBinarySection(Section.getRawDataRefImpl());
}

const BinaryRelocation &
BinaryObjectFile::getBinaryRelocation(const RelocationRef &Ref) const {
  return getBinaryRelocation(Ref.getRawDataRefImpl());
}

const BinaryRelocation &
BinaryObjectFile::getBinaryRelocation(DataRefImpl Ref) const {
  assert(Ref.d.a < Sections.size());
  const BinarySection &Sec = Sections[Ref.d.a];
  assert(Ref.d.b < Sec.Relocations.size());
  return Sec.Relocations[Ref.d.b];
}

Expected<StringRef> BinaryObjectFile::getSymbolName(DataRefImpl Symb) const {
  return getBinarySymbol(Symb).Name;
}

Expected<uint32_t> BinaryObjectFile::getSymbolFlags(DataRefImpl Symb) const {
  return getBinarySymbol(Symb).Flags;
}

uint64_t BinaryObjectFile::getSymbolValueImpl(DataRefImpl Symb) const {
  return getBinarySymbol(Symb).Value;
}

uint64_t BinaryObjectFile::getCommonSymbolSizeImpl(DataRefImpl Symb) const {
  llvm_unreachable("not yet implemented");
  return 0;
}

Expected<SymbolRef::Type>
BinaryObjectFile::getSymbolType(DataRefImpl Symb) const {
  return SymbolRef::ST_Other;
}

Expected<section_iterator>
BinaryObjectFile::getSymbolSection(DataRefImpl Symb) const {
  DataRefImpl Ref;
  Ref.d.a = 0;
  return section_iterator(SectionRef(Ref, this));
}

Expected<uint64_t> BinaryObjectFile::getSymbolAddress(DataRefImpl Sym) const {
  return getSymbolValue(Sym);
}

section_iterator BinaryObjectFile::section_begin() const {
  DataRefImpl Ref;
  Ref.d.a = 0;
  return section_iterator(SectionRef(Ref, this));
}

section_iterator BinaryObjectFile::section_end() const {
  DataRefImpl Ref;
  Ref.d.a = Sections.size();
  return section_iterator(SectionRef(Ref, this));
}
void BinaryObjectFile::moveSectionNext(DataRefImpl &Sec) const { Sec.d.a++; }

Expected<StringRef> BinaryObjectFile::getSectionName(DataRefImpl Ref) const {
  return getBinarySection(Ref).Name;
}

uint64_t BinaryObjectFile::getSectionAddress(DataRefImpl Ref) const {
  return getBinarySection(Ref).Address;
}
uint64_t BinaryObjectFile::getSectionIndex(DataRefImpl Ref) const {
  return getBinarySection(Ref).Index;
}
uint64_t BinaryObjectFile::getSectionSize(DataRefImpl Ref) const {
  return getBinarySection(Ref).Size;
}

Expected<ArrayRef<uint8_t>>
BinaryObjectFile::getSectionContents(DataRefImpl Sec) const {
  return ArrayRef<uint8_t>((const uint8_t *)Data.getBuffer().data(),
                           Data.getBufferSize());
}

uint64_t BinaryObjectFile::getSectionAlignment(DataRefImpl Sec) const {
  return 1;
}

bool BinaryObjectFile::isSectionCompressed(DataRefImpl Sec) const {
  return false;
}
bool BinaryObjectFile::isSectionText(DataRefImpl Sec) const { return true; }
bool BinaryObjectFile::isSectionData(DataRefImpl Sec) const { return false; }
bool BinaryObjectFile::isSectionBSS(DataRefImpl Sec) const { return false; }
bool BinaryObjectFile::isSectionVirtual(DataRefImpl Sec) const { return false; }

relocation_iterator BinaryObjectFile::section_rel_begin(DataRefImpl Ref) const {
  DataRefImpl RelocRef;
  RelocRef.d.a = Ref.d.a;
  RelocRef.d.b = 0;
  return relocation_iterator(RelocationRef(RelocRef, this));
}

relocation_iterator BinaryObjectFile::section_rel_end(DataRefImpl Ref) const {
  const BinarySection &Sec = getBinarySection(Ref);
  DataRefImpl RelocRef;
  RelocRef.d.a = Ref.d.a;
  RelocRef.d.b = Sec.Relocations.size();
  return relocation_iterator(RelocationRef(RelocRef, this));
}

void BinaryObjectFile::moveRelocationNext(DataRefImpl &Rel) const { Rel.d.b++; }

uint64_t BinaryObjectFile::getRelocationOffset(DataRefImpl Ref) const {
  const BinaryRelocation &Rel = getBinaryRelocation(Ref);
  return Rel.Offset;
}

symbol_iterator BinaryObjectFile::getRelocationSymbol(DataRefImpl Ref) const {
  const BinaryRelocation &Rel = getBinaryRelocation(Ref);
  DataRefImpl Sym;
  Sym.d.a = 1;
  Sym.d.b = Rel.Symbol;
  return symbol_iterator(SymbolRef(Sym, this));
}

uint64_t BinaryObjectFile::getRelocationType(DataRefImpl Ref) const {
  const BinaryRelocation &Rel = getBinaryRelocation(Ref);
  return Rel.Type;
}

void BinaryObjectFile::getRelocationTypeName(
    DataRefImpl Ref, SmallVectorImpl<char> &Result) const {
  const BinaryRelocation &Rel = getBinaryRelocation(Ref);
  StringRef Res;
  switch (Rel.Type) {
  case 0:
  default:
    Res = "unknown";
    break;
  }
  Result.append(Res.begin(), Res.end());
}

uint8_t BinaryObjectFile::getBytesInAddress() const {
  return is64Bit() ? 8 : 4;
}

StringRef BinaryObjectFile::getFileFormatName() const { return "binary"; }

Triple::ArchType BinaryObjectFile::getArch() const {
  return Triple::UnknownArch;
}

Expected<SubtargetFeatures> BinaryObjectFile::getFeatures() const {
  return SubtargetFeatures();
}

std::optional<StringRef> BinaryObjectFile::tryGetCPUName() const {
  return std::nullopt;
}

bool BinaryObjectFile::isRelocatableObject() const { return false; }
