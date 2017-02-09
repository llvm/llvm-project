//===- WasmObjectFile.h - Wasm object file implementation -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the WasmObjectFile class, which implements the ObjectFile
// interface for Wasm files.
//
// See: https://github.com/WebAssembly/design/blob/master/BinaryEncoding.md
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_WASM_H
#define LLVM_OBJECT_WASM_H

#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Wasm.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace llvm {
namespace object {

class WasmObjectFile : public ObjectFile {
public:
  WasmObjectFile(MemoryBufferRef Object, Error &Err);

  const wasm::WasmObjectHeader &getHeader() const;
  const wasm::WasmSection *getWasmSection(const SectionRef &Section) const;

  static bool classof(const Binary *v) { return v->isWasm(); }

protected:
  void moveSymbolNext(DataRefImpl &Symb) const override;

  std::error_code printSymbolName(raw_ostream &OS,
                                  DataRefImpl Symb) const override;

  uint32_t getSymbolFlags(DataRefImpl Symb) const override;

  basic_symbol_iterator symbol_begin() const override;

  basic_symbol_iterator symbol_end() const override;
  Expected<StringRef> getSymbolName(DataRefImpl Symb) const override;

  Expected<uint64_t> getSymbolAddress(DataRefImpl Symb) const override;
  uint64_t getSymbolValueImpl(DataRefImpl Symb) const override;
  uint32_t getSymbolAlignment(DataRefImpl Symb) const override;
  uint64_t getCommonSymbolSizeImpl(DataRefImpl Symb) const override;
  Expected<SymbolRef::Type> getSymbolType(DataRefImpl Symb) const override;
  Expected<section_iterator> getSymbolSection(DataRefImpl Symb) const override;

  // Overrides from SectionRef.
  void moveSectionNext(DataRefImpl &Sec) const override;
  std::error_code getSectionName(DataRefImpl Sec,
                                 StringRef &Res) const override;
  uint64_t getSectionAddress(DataRefImpl Sec) const override;
  uint64_t getSectionSize(DataRefImpl Sec) const override;
  std::error_code getSectionContents(DataRefImpl Sec,
                                     StringRef &Res) const override;
  uint64_t getSectionAlignment(DataRefImpl Sec) const override;
  bool isSectionCompressed(DataRefImpl Sec) const override;
  bool isSectionText(DataRefImpl Sec) const override;
  bool isSectionData(DataRefImpl Sec) const override;
  bool isSectionBSS(DataRefImpl Sec) const override;
  bool isSectionVirtual(DataRefImpl Sec) const override;
  bool isSectionBitcode(DataRefImpl Sec) const override;
  relocation_iterator section_rel_begin(DataRefImpl Sec) const override;
  relocation_iterator section_rel_end(DataRefImpl Sec) const override;
  section_iterator getRelocatedSection(DataRefImpl Sec) const override;

  // Overrides from RelocationRef.
  void moveRelocationNext(DataRefImpl &Rel) const override;
  uint64_t getRelocationOffset(DataRefImpl Rel) const override;
  symbol_iterator getRelocationSymbol(DataRefImpl Rel) const override;
  uint64_t getRelocationType(DataRefImpl Rel) const override;
  void getRelocationTypeName(DataRefImpl Rel,
                             SmallVectorImpl<char> &Result) const override;

  section_iterator section_begin() const override;
  section_iterator section_end() const override;
  uint8_t getBytesInAddress() const override;
  StringRef getFileFormatName() const override;
  unsigned getArch() const override;
  SubtargetFeatures getFeatures() const override;
  bool isRelocatableObject() const override;

private:
  const uint8_t *getPtr(size_t Offset) const;
  Error parseCustomSection(wasm::WasmSection &Sec, const uint8_t *Ptr,
                           size_t Length);

  wasm::WasmObjectHeader Header;
  std::vector<wasm::WasmSection> Sections;
};

} // end namespace object
} // end namespace llvm

#endif // LLVM_OBJECT_WASM_H
