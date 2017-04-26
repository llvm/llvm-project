//===- WasmYAML.h - Wasm YAMLIO implementation ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file declares classes for handling the YAML representation
/// of wasm binaries.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECTYAML_WASMYAML_H
#define LLVM_OBJECTYAML_WASMYAML_H

#include "llvm/ObjectYAML/YAML.h"
#include "llvm/Support/Wasm.h"

namespace llvm {
namespace WasmYAML {

LLVM_YAML_STRONG_TYPEDEF(uint32_t, SectionType)
LLVM_YAML_STRONG_TYPEDEF(int32_t, ValueType)
LLVM_YAML_STRONG_TYPEDEF(int32_t, TableType)
LLVM_YAML_STRONG_TYPEDEF(int32_t, SignatureForm)
LLVM_YAML_STRONG_TYPEDEF(uint32_t, ExportKind)
LLVM_YAML_STRONG_TYPEDEF(uint32_t, Opcode)
LLVM_YAML_STRONG_TYPEDEF(uint32_t, RelocType)

struct FileHeader {
  yaml::Hex32 Version;
};

struct Import {
  StringRef Module;
  StringRef Field;
  ExportKind Kind;
  union {
    uint32_t SigIndex;
    ValueType GlobalType;
  };
  bool GlobalMutable;
};

struct Limits {
  yaml::Hex32 Flags;
  yaml::Hex32 Initial;
  yaml::Hex32 Maximum;
};

struct Table {
  TableType ElemType;
  Limits TableLimits;
};

struct Export {
  StringRef Name;
  ExportKind Kind;
  uint32_t Index;
};

struct ElemSegment {
  uint32_t TableIndex;
  wasm::WasmInitExpr Offset;
  std::vector<uint32_t> Functions;
};

struct Global {
  ValueType Type;
  bool Mutable;
  wasm::WasmInitExpr InitExpr;
};

struct LocalDecl {
  ValueType Type;
  uint32_t Count;
};

struct Function {
  std::vector<LocalDecl> Locals;
  yaml::BinaryRef Body;
};

struct Relocation {
  RelocType Type;
  uint32_t Index;
  yaml::Hex32 Offset;
  int32_t Addend;
};

struct DataSegment {
  uint32_t Index;
  wasm::WasmInitExpr Offset;
  yaml::BinaryRef Content;
};

struct Signature {
  Signature() : Form(wasm::WASM_TYPE_FUNC) {}

  uint32_t Index;
  SignatureForm Form;
  std::vector<ValueType> ParamTypes;
  ValueType ReturnType;
};

struct Section {
  Section(SectionType SecType) : Type(SecType) {}
  virtual ~Section();

  SectionType Type;
  std::vector<Relocation> Relocations;
};

struct CustomSection : Section {
  CustomSection() : Section(wasm::WASM_SEC_CUSTOM) {}
  static bool classof(const Section *S) {
    return S->Type == wasm::WASM_SEC_CUSTOM;
  }

  StringRef Name;
  yaml::BinaryRef Payload;
};

struct TypeSection : Section {
  TypeSection() : Section(wasm::WASM_SEC_TYPE) {}
  static bool classof(const Section *S) {
    return S->Type == wasm::WASM_SEC_TYPE;
  }

  std::vector<Signature> Signatures;
};

struct ImportSection : Section {
  ImportSection() : Section(wasm::WASM_SEC_IMPORT) {}
  static bool classof(const Section *S) {
    return S->Type == wasm::WASM_SEC_IMPORT;
  }

  std::vector<Import> Imports;
};

struct FunctionSection : Section {
  FunctionSection() : Section(wasm::WASM_SEC_FUNCTION) {}
  static bool classof(const Section *S) {
    return S->Type == wasm::WASM_SEC_FUNCTION;
  }

  std::vector<uint32_t> FunctionTypes;
};

struct TableSection : Section {
  TableSection() : Section(wasm::WASM_SEC_TABLE) {}
  static bool classof(const Section *S) {
    return S->Type == wasm::WASM_SEC_TABLE;
  }

  std::vector<Table> Tables;
};

struct MemorySection : Section {
  MemorySection() : Section(wasm::WASM_SEC_MEMORY) {}
  static bool classof(const Section *S) {
    return S->Type == wasm::WASM_SEC_MEMORY;
  }

  std::vector<Limits> Memories;
};

struct GlobalSection : Section {
  GlobalSection() : Section(wasm::WASM_SEC_GLOBAL) {}
  static bool classof(const Section *S) {
    return S->Type == wasm::WASM_SEC_GLOBAL;
  }

  std::vector<Global> Globals;
};

struct ExportSection : Section {
  ExportSection() : Section(wasm::WASM_SEC_EXPORT) {}
  static bool classof(const Section *S) {
    return S->Type == wasm::WASM_SEC_EXPORT;
  }

  std::vector<Export> Exports;
};

struct StartSection : Section {
  StartSection() : Section(wasm::WASM_SEC_START) {}
  static bool classof(const Section *S) {
    return S->Type == wasm::WASM_SEC_START;
  }

  uint32_t StartFunction;
};

struct ElemSection : Section {
  ElemSection() : Section(wasm::WASM_SEC_ELEM) {}
  static bool classof(const Section *S) {
    return S->Type == wasm::WASM_SEC_ELEM;
  }

  std::vector<ElemSegment> Segments;
};

struct CodeSection : Section {
  CodeSection() : Section(wasm::WASM_SEC_CODE) {}
  static bool classof(const Section *S) {
    return S->Type == wasm::WASM_SEC_CODE;
  }

  std::vector<Function> Functions;
};

struct DataSection : Section {
  DataSection() : Section(wasm::WASM_SEC_DATA) {}
  static bool classof(const Section *S) {
    return S->Type == wasm::WASM_SEC_DATA;
  }

  std::vector<DataSegment> Segments;
};

struct Object {
  FileHeader Header;
  std::vector<std::unique_ptr<Section>> Sections;
};

} // end namespace WasmYAML
} // end namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(std::unique_ptr<llvm::WasmYAML::Section>)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::WasmYAML::Signature)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::WasmYAML::ValueType)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::WasmYAML::Table)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::WasmYAML::Import)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::WasmYAML::Export)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::WasmYAML::ElemSegment)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::WasmYAML::Limits)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::WasmYAML::DataSegment)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::WasmYAML::Global)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::WasmYAML::Function)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::WasmYAML::LocalDecl)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::WasmYAML::Relocation)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(uint32_t)

namespace llvm {
namespace yaml {

template <> struct MappingTraits<WasmYAML::FileHeader> {
  static void mapping(IO &IO, WasmYAML::FileHeader &FileHdr);
};

template <> struct MappingTraits<std::unique_ptr<WasmYAML::Section>> {
  static void mapping(IO &IO, std::unique_ptr<WasmYAML::Section> &Section);
};

template <> struct MappingTraits<WasmYAML::Object> {
  static void mapping(IO &IO, WasmYAML::Object &Object);
};

template <> struct MappingTraits<WasmYAML::Import> {
  static void mapping(IO &IO, WasmYAML::Import &Import);
};

template <> struct MappingTraits<WasmYAML::Export> {
  static void mapping(IO &IO, WasmYAML::Export &Export);
};

template <> struct MappingTraits<WasmYAML::Global> {
  static void mapping(IO &IO, WasmYAML::Global &Global);
};

template <> struct ScalarEnumerationTraits<WasmYAML::SectionType> {
  static void enumeration(IO &IO, WasmYAML::SectionType &Type);
};

template <> struct MappingTraits<WasmYAML::Signature> {
  static void mapping(IO &IO, WasmYAML::Signature &Signature);
};

template <> struct MappingTraits<WasmYAML::Table> {
  static void mapping(IO &IO, WasmYAML::Table &Table);
};

template <> struct MappingTraits<WasmYAML::Limits> {
  static void mapping(IO &IO, WasmYAML::Limits &Limits);
};

template <> struct MappingTraits<WasmYAML::Function> {
  static void mapping(IO &IO, WasmYAML::Function &Function);
};

template <> struct MappingTraits<WasmYAML::Relocation> {
  static void mapping(IO &IO, WasmYAML::Relocation &Relocation);
};

template <> struct MappingTraits<WasmYAML::LocalDecl> {
  static void mapping(IO &IO, WasmYAML::LocalDecl &LocalDecl);
};

template <> struct MappingTraits<wasm::WasmInitExpr> {
  static void mapping(IO &IO, wasm::WasmInitExpr &Expr);
};

template <> struct MappingTraits<WasmYAML::DataSegment> {
  static void mapping(IO &IO, WasmYAML::DataSegment &Segment);
};

template <> struct MappingTraits<WasmYAML::ElemSegment> {
  static void mapping(IO &IO, WasmYAML::ElemSegment &Segment);
};

template <> struct ScalarEnumerationTraits<WasmYAML::ValueType> {
  static void enumeration(IO &IO, WasmYAML::ValueType &Type);
};

template <> struct ScalarEnumerationTraits<WasmYAML::ExportKind> {
  static void enumeration(IO &IO, WasmYAML::ExportKind &Kind);
};

template <> struct ScalarEnumerationTraits<WasmYAML::TableType> {
  static void enumeration(IO &IO, WasmYAML::TableType &Type);
};

template <> struct ScalarEnumerationTraits<WasmYAML::Opcode> {
  static void enumeration(IO &IO, WasmYAML::Opcode &Opcode);
};

template <> struct ScalarEnumerationTraits<WasmYAML::RelocType> {
  static void enumeration(IO &IO, WasmYAML::RelocType &Kind);
};

} // end namespace yaml
} // end namespace llvm

#endif
