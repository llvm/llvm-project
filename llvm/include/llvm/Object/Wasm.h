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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
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

class WasmSymbol {
public:
  enum class SymbolType {
    FUNCTION_IMPORT,
    FUNCTION_EXPORT,
    GLOBAL_IMPORT,
    GLOBAL_EXPORT,
    DEBUG_FUNCTION_NAME,
  };

  WasmSymbol(StringRef Name, SymbolType Type, uint32_t Section,
             uint32_t ElementIndex)
      : Name(Name), Type(Type), Section(Section), ElementIndex(ElementIndex) {}

  StringRef Name;
  SymbolType Type;
  uint32_t Section;
  uint32_t ElementIndex;
};

class WasmSection {
public:
  WasmSection() = default;

  uint32_t Type = 0; // Section type (See below)
  uint32_t Offset = 0; // Offset with in the file
  StringRef Name; // Section name (User-defined sections only)
  ArrayRef<uint8_t> Content; // Section content
  std::vector<wasm::WasmRelocation> Relocations; // Relocations for this section
};

class WasmObjectFile : public ObjectFile {
public:
  WasmObjectFile(MemoryBufferRef Object, Error &Err);

  const wasm::WasmObjectHeader &getHeader() const;
  const WasmSymbol &getWasmSymbol(const DataRefImpl &Symb) const;
  const WasmSymbol &getWasmSymbol(const SymbolRef &Symbol) const;
  const WasmSection &getWasmSection(const SectionRef &Section) const;
  const wasm::WasmRelocation &getWasmRelocation(const RelocationRef& Ref) const;

  static bool classof(const Binary *v) { return v->isWasm(); }

  const std::vector<wasm::WasmSignature>& types() const { return Signatures; }
  const std::vector<uint32_t>& functionTypes() const { return FunctionTypes; }
  const std::vector<wasm::WasmImport>& imports() const { return Imports; }
  const std::vector<wasm::WasmTable>& tables() const { return Tables; }
  const std::vector<wasm::WasmLimits>& memories() const { return Memories; }
  const std::vector<wasm::WasmGlobal>& globals() const { return Globals; }
  const std::vector<wasm::WasmExport>& exports() const { return Exports; }

  uint32_t getNumberOfSymbols() const {
    return Symbols.size();
  }

  const std::vector<wasm::WasmElemSegment>& elements() const {
    return ElemSegments;
  }

  const std::vector<wasm::WasmDataSegment>& dataSegments() const {
    return DataSegments;
  }

  const std::vector<wasm::WasmFunction>& functions() const { return Functions; }
  const ArrayRef<uint8_t>& code() const { return CodeSection; }
  uint32_t startFunction() const { return StartFunction; }

  void moveSymbolNext(DataRefImpl &Symb) const override;

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
  uint64_t getSectionIndex(DataRefImpl Sec) const override;
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
  const WasmSection &getWasmSection(DataRefImpl Ref) const;
  const wasm::WasmRelocation &getWasmRelocation(DataRefImpl Ref) const;

  WasmSection* findCustomSectionByName(StringRef Name);
  WasmSection* findSectionByType(uint32_t Type);

  const uint8_t *getPtr(size_t Offset) const;
  Error parseSection(WasmSection &Sec);
  Error parseCustomSection(WasmSection &Sec, const uint8_t *Ptr,
                           const uint8_t *End);

  // Standard section types
  Error parseTypeSection(const uint8_t *Ptr, const uint8_t *End);
  Error parseImportSection(const uint8_t *Ptr, const uint8_t *End);
  Error parseFunctionSection(const uint8_t *Ptr, const uint8_t *End);
  Error parseTableSection(const uint8_t *Ptr, const uint8_t *End);
  Error parseMemorySection(const uint8_t *Ptr, const uint8_t *End);
  Error parseGlobalSection(const uint8_t *Ptr, const uint8_t *End);
  Error parseExportSection(const uint8_t *Ptr, const uint8_t *End);
  Error parseStartSection(const uint8_t *Ptr, const uint8_t *End);
  Error parseElemSection(const uint8_t *Ptr, const uint8_t *End);
  Error parseCodeSection(const uint8_t *Ptr, const uint8_t *End);
  Error parseDataSection(const uint8_t *Ptr, const uint8_t *End);

  // Custom section types
  Error parseNameSection(const uint8_t *Ptr, const uint8_t *End);
  Error parseRelocSection(StringRef Name, const uint8_t *Ptr,
                          const uint8_t *End);

  wasm::WasmObjectHeader Header;
  std::vector<WasmSection> Sections;
  std::vector<wasm::WasmSignature> Signatures;
  std::vector<uint32_t> FunctionTypes;
  std::vector<wasm::WasmTable> Tables;
  std::vector<wasm::WasmLimits> Memories;
  std::vector<wasm::WasmGlobal> Globals;
  std::vector<wasm::WasmImport> Imports;
  std::vector<wasm::WasmExport> Exports;
  std::vector<wasm::WasmElemSegment> ElemSegments;
  std::vector<wasm::WasmDataSegment> DataSegments;
  std::vector<WasmSymbol> Symbols;
  std::vector<wasm::WasmFunction> Functions;
  ArrayRef<uint8_t> CodeSection;
  uint32_t StartFunction = -1;
};

} // end namespace object
} // end namespace llvm

#endif // LLVM_OBJECT_WASM_H
