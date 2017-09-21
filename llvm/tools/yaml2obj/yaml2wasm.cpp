//===- yaml2wasm - Convert YAML to a Wasm object file --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief The Wasm component of yaml2obj.
///
//===----------------------------------------------------------------------===//
//
#include "yaml2obj.h"
#include "llvm/Object/Wasm.h"
#include "llvm/ObjectYAML/ObjectYAML.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/LEB128.h"

using namespace llvm;

/// This parses a yaml stream that represents a Wasm object file.
/// See docs/yaml2obj for the yaml scheema.
class WasmWriter {
public:
  WasmWriter(WasmYAML::Object &Obj) : Obj(Obj) {}
  int writeWasm(raw_ostream &OS);

private:
  int writeRelocSection(raw_ostream &OS, WasmYAML::Section &Sec);

  int writeSectionContent(raw_ostream &OS, WasmYAML::CustomSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::TypeSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::ImportSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::FunctionSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::TableSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::MemorySection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::GlobalSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::ExportSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::StartSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::ElemSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::CodeSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::DataSection &Section);

  // Custom section types
  int writeSectionContent(raw_ostream &OS, WasmYAML::NameSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::LinkingSection &Section);
  WasmYAML::Object &Obj;
};

static int writeUint64(raw_ostream &OS, uint64_t Value) {
  char Data[sizeof(Value)];
  support::endian::write64le(Data, Value);
  OS.write(Data, sizeof(Data));
  return 0;
}

static int writeUint32(raw_ostream &OS, uint32_t Value) {
  char Data[sizeof(Value)];
  support::endian::write32le(Data, Value);
  OS.write(Data, sizeof(Data));
  return 0;
}

static int writeUint8(raw_ostream &OS, uint8_t Value) {
  char Data[sizeof(Value)];
  memcpy(Data, &Value, sizeof(Data));
  OS.write(Data, sizeof(Data));
  return 0;
}

static int writeStringRef(const StringRef &Str, raw_ostream &OS) {
  encodeULEB128(Str.size(), OS);
  OS << Str;
  return 0;
}

static int writeLimits(const WasmYAML::Limits &Lim, raw_ostream &OS) {
  encodeULEB128(Lim.Flags, OS);
  encodeULEB128(Lim.Initial, OS);
  if (Lim.Flags & wasm::WASM_LIMITS_FLAG_HAS_MAX)
    encodeULEB128(Lim.Maximum, OS);
  return 0;
}

static int writeInitExpr(const wasm::WasmInitExpr &InitExpr, raw_ostream &OS) {
  writeUint8(OS, InitExpr.Opcode);
  switch (InitExpr.Opcode) {
  case wasm::WASM_OPCODE_I32_CONST:
    encodeSLEB128(InitExpr.Value.Int32, OS);
    break;
  case wasm::WASM_OPCODE_I64_CONST:
    encodeSLEB128(InitExpr.Value.Int64, OS);
    break;
  case wasm::WASM_OPCODE_F32_CONST:
    writeUint32(OS, InitExpr.Value.Float32);
    break;
  case wasm::WASM_OPCODE_F64_CONST:
    writeUint64(OS, InitExpr.Value.Float64);
    break;
  case wasm::WASM_OPCODE_GET_GLOBAL:
    encodeULEB128(InitExpr.Value.Global, OS);
    break;
  default:
    errs() << "Unknown opcode in init_expr: " << InitExpr.Opcode;
    return 1;
  }
  writeUint8(OS, wasm::WASM_OPCODE_END);
  return 0;
}

class SubSectionWriter {
  raw_ostream &OS;
  std::string OutString;
  raw_string_ostream StringStream;

public:
  SubSectionWriter(raw_ostream &OS) : OS(OS), StringStream(OutString) {}

  void Done() {
    StringStream.flush();
    encodeULEB128(OutString.size(), OS);
    OS << OutString;
    OutString.clear();
  }

  raw_ostream& GetStream() {
    return StringStream;
  }
};

int WasmWriter::writeSectionContent(raw_ostream &OS, WasmYAML::LinkingSection &Section) {
  writeStringRef(Section.Name, OS);

  SubSectionWriter SubSection(OS);

  // DATA_SIZE subsection
  encodeULEB128(wasm::WASM_DATA_SIZE, OS);
  encodeULEB128(Section.DataSize, SubSection.GetStream());
  SubSection.Done();

  // DATA_ALIGNMENT subsection
  encodeULEB128(wasm::WASM_DATA_ALIGNMENT, OS);
  encodeULEB128(Section.DataAlignment, SubSection.GetStream());
  SubSection.Done();

  // SYMBOL_INFO subsection
  if (Section.SymbolInfos.size()) {
    encodeULEB128(wasm::WASM_SYMBOL_INFO, OS);

    encodeULEB128(Section.SymbolInfos.size(), SubSection.GetStream());
    for (const WasmYAML::SymbolInfo &Info : Section.SymbolInfos) {
      writeStringRef(Info.Name, SubSection.GetStream());
      encodeULEB128(Info.Flags, SubSection.GetStream());
    }

    SubSection.Done();
  }

  // SEGMENT_NAMES subsection
  if (Section.SegmentNames.size()) {
    encodeULEB128(wasm::WASM_SEGMENT_NAMES, OS);
    encodeULEB128(Section.SegmentNames.size(), SubSection.GetStream());
    for (const WasmYAML::NameEntry &NameEntry : Section.SegmentNames) {
      encodeULEB128(NameEntry.Index, SubSection.GetStream());
      writeStringRef(NameEntry.Name, SubSection.GetStream());
    }
    SubSection.Done();
  }
  return 0;
}

int WasmWriter::writeSectionContent(raw_ostream &OS, WasmYAML::NameSection &Section) {
  writeStringRef(Section.Name, OS);
  if (Section.FunctionNames.size()) {
    encodeULEB128(wasm::WASM_NAMES_FUNCTION, OS);

    SubSectionWriter SubSection(OS);

    encodeULEB128(Section.FunctionNames.size(), SubSection.GetStream());
    for (const WasmYAML::NameEntry &NameEntry : Section.FunctionNames) {
      encodeULEB128(NameEntry.Index, SubSection.GetStream());
      writeStringRef(NameEntry.Name, SubSection.GetStream());
    }

    SubSection.Done();
  }
  return 0;
}

int WasmWriter::writeSectionContent(raw_ostream &OS,
                                    WasmYAML::CustomSection &Section) {
  if (auto S = dyn_cast<WasmYAML::NameSection>(&Section)) {
    if (auto Err = writeSectionContent(OS, *S))
      return Err;
  } else if (auto S = dyn_cast<WasmYAML::LinkingSection>(&Section)) {
    if (auto Err = writeSectionContent(OS, *S))
      return Err;
  } else {
    Section.Payload.writeAsBinary(OS);
  }
  return 0;
}

int WasmWriter::writeSectionContent(raw_ostream &OS,
                                    WasmYAML::TypeSection &Section) {
  encodeULEB128(Section.Signatures.size(), OS);
  for (const WasmYAML::Signature &Sig : Section.Signatures) {
    encodeSLEB128(Sig.Form, OS);
    encodeULEB128(Sig.ParamTypes.size(), OS);
    for (auto ParamType : Sig.ParamTypes)
      encodeSLEB128(ParamType, OS);
    if (Sig.ReturnType == wasm::WASM_TYPE_NORESULT) {
      encodeSLEB128(0, OS);
    } else {
      encodeULEB128(1, OS);
      encodeSLEB128(Sig.ReturnType, OS);
    }
  }
  return 0;
}

int WasmWriter::writeSectionContent(raw_ostream &OS,
                                    WasmYAML::ImportSection &Section) {
  encodeULEB128(Section.Imports.size(), OS);
  for (const WasmYAML::Import &Import : Section.Imports) {
    writeStringRef(Import.Module, OS);
    writeStringRef(Import.Field, OS);
    encodeULEB128(Import.Kind, OS);
    switch (Import.Kind) {
    case wasm::WASM_EXTERNAL_FUNCTION:
      encodeULEB128(Import.SigIndex, OS);
      break;
    case wasm::WASM_EXTERNAL_GLOBAL:
      encodeSLEB128(Import.GlobalImport.Type, OS);
      writeUint8(OS, Import.GlobalImport.Mutable);
      break;
    case wasm::WASM_EXTERNAL_MEMORY:
      writeLimits(Import.Memory, OS);
      break;
    case wasm::WASM_EXTERNAL_TABLE:
      encodeSLEB128(Import.TableImport.ElemType, OS);
      writeLimits(Import.TableImport.TableLimits, OS);
      break;
    default:
      errs() << "Unknown import type: " << Import.Kind;
      return 1;
    }
  }
  return 0;
}

int WasmWriter::writeSectionContent(raw_ostream &OS,
                                    WasmYAML::FunctionSection &Section) {
  encodeULEB128(Section.FunctionTypes.size(), OS);
  for (uint32_t FuncType : Section.FunctionTypes) {
    encodeULEB128(FuncType, OS);
  }
  return 0;
}

int WasmWriter::writeSectionContent(raw_ostream &OS,
                                    WasmYAML::ExportSection &Section) {
  encodeULEB128(Section.Exports.size(), OS);
  for (const WasmYAML::Export &Export : Section.Exports) {
    writeStringRef(Export.Name, OS);
    encodeULEB128(Export.Kind, OS);
    encodeULEB128(Export.Index, OS);
  }
  return 0;
}

int WasmWriter::writeSectionContent(raw_ostream &OS,
                                    WasmYAML::StartSection &Section) {
  encodeULEB128(Section.StartFunction, OS);
  return 0;
}

int WasmWriter::writeSectionContent(raw_ostream &OS,
                                    WasmYAML::TableSection &Section) {
  encodeULEB128(Section.Tables.size(), OS);
  for (auto &Table : Section.Tables) {
    encodeSLEB128(Table.ElemType, OS);
    writeLimits(Table.TableLimits, OS);
  }
  return 0;
}

int WasmWriter::writeSectionContent(raw_ostream &OS,
                                    WasmYAML::MemorySection &Section) {
  encodeULEB128(Section.Memories.size(), OS);
  for (const WasmYAML::Limits &Mem : Section.Memories) {
    writeLimits(Mem, OS);
  }
  return 0;
}

int WasmWriter::writeSectionContent(raw_ostream &OS,
                                    WasmYAML::GlobalSection &Section) {
  encodeULEB128(Section.Globals.size(), OS);
  for (auto &Global : Section.Globals) {
    encodeSLEB128(Global.Type, OS);
    writeUint8(OS, Global.Mutable);
    writeInitExpr(Global.InitExpr, OS);
  }
  return 0;
}

int WasmWriter::writeSectionContent(raw_ostream &OS,
                                    WasmYAML::ElemSection &Section) {
  encodeULEB128(Section.Segments.size(), OS);
  for (auto &Segment : Section.Segments) {
    encodeULEB128(Segment.TableIndex, OS);
    writeInitExpr(Segment.Offset, OS);

    encodeULEB128(Segment.Functions.size(), OS);
    for (auto &Function : Segment.Functions) {
      encodeULEB128(Function, OS);
    }
  }
  return 0;
}

int WasmWriter::writeSectionContent(raw_ostream &OS,
                                    WasmYAML::CodeSection &Section) {
  encodeULEB128(Section.Functions.size(), OS);
  for (auto &Func : Section.Functions) {
    std::string OutString;
    raw_string_ostream StringStream(OutString);

    encodeULEB128(Func.Locals.size(), StringStream);
    for (auto &LocalDecl : Func.Locals) {
      encodeULEB128(LocalDecl.Count, StringStream);
      encodeSLEB128(LocalDecl.Type, StringStream);
    }

    Func.Body.writeAsBinary(StringStream);

    // Write the section size followed by the content
    StringStream.flush();
    encodeULEB128(OutString.size(), OS);
    OS << OutString;
  }
  return 0;
}

int WasmWriter::writeSectionContent(raw_ostream &OS,
                                    WasmYAML::DataSection &Section) {
  encodeULEB128(Section.Segments.size(), OS);
  for (auto &Segment : Section.Segments) {
    encodeULEB128(Segment.MemoryIndex, OS);
    writeInitExpr(Segment.Offset, OS);
    encodeULEB128(Segment.Content.binary_size(), OS);
    Segment.Content.writeAsBinary(OS);
  }
  return 0;
}

int WasmWriter::writeRelocSection(raw_ostream &OS,
                                  WasmYAML::Section &Sec) {
  StringRef Name;
  switch (Sec.Type) {
    case wasm::WASM_SEC_CODE:
      Name = "reloc.CODE";
      break;
    case wasm::WASM_SEC_DATA:
      Name = "reloc.DATA";
      break;
    default:
      llvm_unreachable("not yet implemented");
      return 1;
  }

  writeStringRef(Name, OS);
  encodeULEB128(Sec.Type, OS);
  encodeULEB128(Sec.Relocations.size(), OS);

  for (auto Reloc: Sec.Relocations) {
    encodeULEB128(Reloc.Type, OS);
    encodeULEB128(Reloc.Offset, OS);
    encodeULEB128(Reloc.Index, OS);
    switch (Reloc.Type) {
      case wasm::R_WEBASSEMBLY_MEMORY_ADDR_LEB:
      case wasm::R_WEBASSEMBLY_MEMORY_ADDR_SLEB:
      case wasm::R_WEBASSEMBLY_MEMORY_ADDR_I32:
        encodeULEB128(Reloc.Addend, OS);
    }
  }
  return 0;
}


int WasmWriter::writeWasm(raw_ostream &OS) {
  // Write headers
  OS.write(wasm::WasmMagic, sizeof(wasm::WasmMagic));
  writeUint32(OS, Obj.Header.Version);

  // Write each section
  for (const std::unique_ptr<WasmYAML::Section> &Sec : Obj.Sections) {
    encodeULEB128(Sec->Type, OS);

    std::string OutString;
    raw_string_ostream StringStream(OutString);
    if (auto S = dyn_cast<WasmYAML::CustomSection>(Sec.get())) {
      if (auto Err = writeSectionContent(StringStream, *S))
        return Err;
    } else if (auto S = dyn_cast<WasmYAML::TypeSection>(Sec.get())) {
      if (auto Err = writeSectionContent(StringStream, *S))
        return Err;
    } else if (auto S = dyn_cast<WasmYAML::ImportSection>(Sec.get())) {
      if (auto Err = writeSectionContent(StringStream, *S))
        return Err;
    } else if (auto S = dyn_cast<WasmYAML::FunctionSection>(Sec.get())) {
      if (auto Err = writeSectionContent(StringStream, *S))
        return Err;
    } else if (auto S = dyn_cast<WasmYAML::TableSection>(Sec.get())) {
      if (auto Err = writeSectionContent(StringStream, *S))
        return Err;
    } else if (auto S = dyn_cast<WasmYAML::MemorySection>(Sec.get())) {
      if (auto Err = writeSectionContent(StringStream, *S))
        return Err;
    } else if (auto S = dyn_cast<WasmYAML::GlobalSection>(Sec.get())) {
      if (auto Err = writeSectionContent(StringStream, *S))
        return Err;
    } else if (auto S = dyn_cast<WasmYAML::ExportSection>(Sec.get())) {
      if (auto Err = writeSectionContent(StringStream, *S))
        return Err;
    } else if (auto S = dyn_cast<WasmYAML::StartSection>(Sec.get())) {
      if (auto Err = writeSectionContent(StringStream, *S))
        return Err;
    } else if (auto S = dyn_cast<WasmYAML::ElemSection>(Sec.get())) {
      if (auto Err = writeSectionContent(StringStream, *S))
        return Err;
    } else if (auto S = dyn_cast<WasmYAML::CodeSection>(Sec.get())) {
      if (auto Err = writeSectionContent(StringStream, *S))
        return Err;
    } else if (auto S = dyn_cast<WasmYAML::DataSection>(Sec.get())) {
      if (auto Err = writeSectionContent(StringStream, *S))
        return Err;
    } else {
      errs() << "Unknown section type: " << Sec->Type << "\n";
      return 1;
    }
    StringStream.flush();

    // Write the section size followed by the content
    encodeULEB128(OutString.size(), OS);
    OS << OutString;
  }

  // write reloc sections for any section that have relocations
  for (const std::unique_ptr<WasmYAML::Section> &Sec : Obj.Sections) {
    if (Sec->Relocations.empty())
      continue;

    encodeULEB128(wasm::WASM_SEC_CUSTOM, OS);
    std::string OutString;
    raw_string_ostream StringStream(OutString);
    writeRelocSection(StringStream, *Sec);
    StringStream.flush();

    encodeULEB128(OutString.size(), OS);
    OS << OutString;
  }

  return 0;
}

int yaml2wasm(llvm::WasmYAML::Object &Doc, raw_ostream &Out) {
  WasmWriter Writer(Doc);

  return Writer.writeWasm(Out);
}
