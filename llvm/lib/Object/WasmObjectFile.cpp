//===- WasmObjectFile.cpp - Wasm object file implementation ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/Wasm.h"
#include <algorithm>
#include <cstdint>
#include <system_error>

using namespace llvm;
using namespace object;

Expected<std::unique_ptr<WasmObjectFile>>
ObjectFile::createWasmObjectFile(MemoryBufferRef Buffer) {
  Error Err = Error::success();
  auto ObjectFile = llvm::make_unique<WasmObjectFile>(Buffer, Err);
  if (Err)
    return std::move(Err);

  return std::move(ObjectFile);
}

static uint32_t readUint32(const uint8_t *&Ptr) {
  uint32_t Result = support::endian::read32le(Ptr);
  Ptr += sizeof(Result);
  return Result;
}

static uint64_t readULEB128(const uint8_t *&Ptr) {
  unsigned Count;
  uint64_t Result = decodeULEB128(Ptr, &Count);
  Ptr += Count;
  return Result;
}

static StringRef readString(const uint8_t *&Ptr) {
  uint32_t StringLen = readULEB128(Ptr);
  StringRef Return = StringRef(reinterpret_cast<const char *>(Ptr), StringLen);
  Ptr += StringLen;
  return Return;
}

static Error readSection(wasm::WasmSection &Section, const uint8_t *&Ptr,
                         const uint8_t *Start) {
  // TODO(sbc): Avoid reading past EOF in the case of malformed files.
  Section.Offset = Ptr - Start;
  Section.Type = readULEB128(Ptr);
  uint32_t Size = readULEB128(Ptr);
  if (Size == 0)
    return make_error<StringError>("Zero length section",
                                   object_error::parse_failed);
  Section.Content = ArrayRef<uint8_t>(Ptr, Size);
  Ptr += Size;
  return Error::success();
}

WasmObjectFile::WasmObjectFile(MemoryBufferRef Buffer, Error &Err)
    : ObjectFile(Binary::ID_Wasm, Buffer) {
  ErrorAsOutParameter ErrAsOutParam(&Err);
  Header.Magic = getData().substr(0, 4);
  if (Header.Magic != StringRef("\0asm", 4)) {
    Err = make_error<StringError>("Bad magic number",
                                  object_error::parse_failed);
    return;
  }
  const uint8_t *Ptr = getPtr(4);
  Header.Version = readUint32(Ptr);
  if (Header.Version != wasm::WasmVersion) {
    Err = make_error<StringError>("Bad version number",
                                  object_error::parse_failed);
    return;
  }

  const uint8_t *Eof = getPtr(getData().size());
  wasm::WasmSection Sec;
  while (Ptr < Eof) {
    if ((Err = readSection(Sec, Ptr, getPtr(0))))
      return;
    if (Sec.Type == wasm::WASM_SEC_CUSTOM) {
      if ((Err =
               parseCustomSection(Sec, Sec.Content.data(), Sec.Content.size())))
        return;
    }
    Sections.push_back(Sec);
  }
}

Error WasmObjectFile::parseCustomSection(wasm::WasmSection &Sec,
                                         const uint8_t *Ptr, size_t Length) {
  Sec.Name = readString(Ptr);
  return Error::success();
}

const uint8_t *WasmObjectFile::getPtr(size_t Offset) const {
  return reinterpret_cast<const uint8_t *>(getData().substr(Offset, 1).data());
}

const wasm::WasmObjectHeader &WasmObjectFile::getHeader() const {
  return Header;
}

void WasmObjectFile::moveSymbolNext(DataRefImpl &Symb) const {
  llvm_unreachable("not yet implemented");
}

std::error_code WasmObjectFile::printSymbolName(raw_ostream &OS,
                                                DataRefImpl Symb) const {
  llvm_unreachable("not yet implemented");
  return object_error::invalid_symbol_index;
}

uint32_t WasmObjectFile::getSymbolFlags(DataRefImpl Symb) const {
  llvm_unreachable("not yet implemented");
  return 0;
}

basic_symbol_iterator WasmObjectFile::symbol_begin() const {
  return BasicSymbolRef(DataRefImpl(), this);
}

basic_symbol_iterator WasmObjectFile::symbol_end() const {
  return BasicSymbolRef(DataRefImpl(), this);
}

Expected<StringRef> WasmObjectFile::getSymbolName(DataRefImpl Symb) const {
  llvm_unreachable("not yet implemented");
  return errorCodeToError(object_error::invalid_symbol_index);
}

Expected<uint64_t> WasmObjectFile::getSymbolAddress(DataRefImpl Symb) const {
  llvm_unreachable("not yet implemented");
  return errorCodeToError(object_error::invalid_symbol_index);
}

uint64_t WasmObjectFile::getSymbolValueImpl(DataRefImpl Symb) const {
  llvm_unreachable("not yet implemented");
  return 0;
}

uint32_t WasmObjectFile::getSymbolAlignment(DataRefImpl Symb) const {
  llvm_unreachable("not yet implemented");
  return 0;
}

uint64_t WasmObjectFile::getCommonSymbolSizeImpl(DataRefImpl Symb) const {
  llvm_unreachable("not yet implemented");
  return 0;
}

Expected<SymbolRef::Type>
WasmObjectFile::getSymbolType(DataRefImpl Symb) const {
  llvm_unreachable("not yet implemented");
  return errorCodeToError(object_error::invalid_symbol_index);
}

Expected<section_iterator>
WasmObjectFile::getSymbolSection(DataRefImpl Symb) const {
  llvm_unreachable("not yet implemented");
  return errorCodeToError(object_error::invalid_symbol_index);
}

void WasmObjectFile::moveSectionNext(DataRefImpl &Sec) const { Sec.d.a++; }

std::error_code WasmObjectFile::getSectionName(DataRefImpl Sec,
                                               StringRef &Res) const {
  const wasm::WasmSection &S = Sections[Sec.d.a];
#define ECase(X)                                                               \
  case wasm::WASM_SEC_##X:                                                     \
    Res = #X;                                                                  \
    break
  switch (S.Type) {
    ECase(TYPE);
    ECase(IMPORT);
    ECase(FUNCTION);
    ECase(TABLE);
    ECase(MEMORY);
    ECase(GLOBAL);
    ECase(EXPORT);
    ECase(START);
    ECase(ELEM);
    ECase(CODE);
    ECase(DATA);
  case wasm::WASM_SEC_CUSTOM:
    Res = S.Name;
    break;
  default:
    return object_error::invalid_section_index;
  }
#undef ECase
  return std::error_code();
}

uint64_t WasmObjectFile::getSectionAddress(DataRefImpl Sec) const { return 0; }

uint64_t WasmObjectFile::getSectionSize(DataRefImpl Sec) const {
  const wasm::WasmSection &S = Sections[Sec.d.a];
  return S.Content.size();
}

std::error_code WasmObjectFile::getSectionContents(DataRefImpl Sec,
                                                   StringRef &Res) const {
  const wasm::WasmSection &S = Sections[Sec.d.a];
  // This will never fail since wasm sections can never be empty (user-sections
  // must have a name and non-user sections each have a defined structure).
  Res = StringRef(reinterpret_cast<const char *>(S.Content.data()),
                  S.Content.size());
  return std::error_code();
}

uint64_t WasmObjectFile::getSectionAlignment(DataRefImpl Sec) const {
  return 1;
}

bool WasmObjectFile::isSectionCompressed(DataRefImpl Sec) const {
  return false;
}

bool WasmObjectFile::isSectionText(DataRefImpl Sec) const {
  const wasm::WasmSection &S = Sections[Sec.d.a];
  return S.Type == wasm::WASM_SEC_CODE;
}

bool WasmObjectFile::isSectionData(DataRefImpl Sec) const {
  const wasm::WasmSection &S = Sections[Sec.d.a];
  return S.Type == wasm::WASM_SEC_DATA;
}

bool WasmObjectFile::isSectionBSS(DataRefImpl Sec) const { return false; }

bool WasmObjectFile::isSectionVirtual(DataRefImpl Sec) const { return false; }

bool WasmObjectFile::isSectionBitcode(DataRefImpl Sec) const { return false; }

relocation_iterator WasmObjectFile::section_rel_begin(DataRefImpl Sec) const {
  llvm_unreachable("not yet implemented");
  RelocationRef Rel;
  return relocation_iterator(Rel);
}

relocation_iterator WasmObjectFile::section_rel_end(DataRefImpl Sec) const {
  llvm_unreachable("not yet implemented");
  RelocationRef Rel;
  return relocation_iterator(Rel);
}

section_iterator WasmObjectFile::getRelocatedSection(DataRefImpl Sec) const {
  llvm_unreachable("not yet implemented");
  SectionRef Ref;
  return section_iterator(Ref);
}

void WasmObjectFile::moveRelocationNext(DataRefImpl &Rel) const {
  llvm_unreachable("not yet implemented");
}

uint64_t WasmObjectFile::getRelocationOffset(DataRefImpl Rel) const {
  llvm_unreachable("not yet implemented");
  return 0;
}

symbol_iterator WasmObjectFile::getRelocationSymbol(DataRefImpl Rel) const {
  llvm_unreachable("not yet implemented");
  SymbolRef Ref;
  return symbol_iterator(Ref);
}

uint64_t WasmObjectFile::getRelocationType(DataRefImpl Rel) const {
  llvm_unreachable("not yet implemented");
  return 0;
}

void WasmObjectFile::getRelocationTypeName(
    DataRefImpl Rel, SmallVectorImpl<char> &Result) const {
  llvm_unreachable("not yet implemented");
}

section_iterator WasmObjectFile::section_begin() const {
  DataRefImpl Ref;
  Ref.d.a = 0;
  return section_iterator(SectionRef(Ref, this));
}

section_iterator WasmObjectFile::section_end() const {
  DataRefImpl Ref;
  Ref.d.a = Sections.size();
  return section_iterator(SectionRef(Ref, this));
}

uint8_t WasmObjectFile::getBytesInAddress() const { return 4; }

StringRef WasmObjectFile::getFileFormatName() const { return "WASM"; }

unsigned WasmObjectFile::getArch() const { return Triple::wasm32; }

SubtargetFeatures WasmObjectFile::getFeatures() const {
  return SubtargetFeatures();
}

bool WasmObjectFile::isRelocatableObject() const { return false; }

const wasm::WasmSection *
WasmObjectFile::getWasmSection(const SectionRef &Section) const {
  return &Sections[Section.getRawDataRefImpl().d.a];
}
