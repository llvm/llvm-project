//===- yaml2wasm - Convert YAML to a Wasm object file --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The Wasm component of yaml2obj.
///
//===----------------------------------------------------------------------===//
//

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
  int writeRelocSection(raw_ostream &OS, WasmYAML::Section &Sec,
                        uint32_t SectionIndex);

  int writeSectionContent(raw_ostream &OS, WasmYAML::CustomSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::TypeSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::ImportSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::FunctionSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::TableSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::MemorySection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::GlobalSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::EventSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::ExportSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::StartSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::ElemSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::CodeSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::DataSection &Section);

  // Custom section types
  int writeSectionContent(raw_ostream &OS, WasmYAML::DylinkSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::NameSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::LinkingSection &Section);
  int writeSectionContent(raw_ostream &OS, WasmYAML::ProducersSection &Section);
  WasmYAML::Object &Obj;
  uint32_t NumImportedFunctions = 0;
  uint32_t NumImportedGlobals = 0;
  uint32_t NumImportedEvents = 0;
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
  writeUint8(OS, Lim.Flags);
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
  case wasm::WASM_OPCODE_GLOBAL_GET:
    encodeULEB128(InitExpr.Value.Global, OS);
    break;
  default:
    errs() << "Unknown opcode in init_expr: " << InitExpr.Opcode << "\n";
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

  void done() {
    StringStream.flush();
    encodeULEB128(OutString.size(), OS);
    OS << OutString;
    OutString.clear();
  }

  raw_ostream &getStream() { return StringStream; }
};

int WasmWriter::writeSectionContent(raw_ostream &OS,
                                    WasmYAML::DylinkSection &Section) {
  writeStringRef(Section.Name, OS);
  encodeULEB128(Section.MemorySize, OS);
  encodeULEB128(Section.MemoryAlignment, OS);
  encodeULEB128(Section.TableSize, OS);
  encodeULEB128(Section.TableAlignment, OS);
  encodeULEB128(Section.Needed.size(), OS);
  for (StringRef Needed : Section.Needed) {
    writeStringRef(Needed, OS);
  }
  return 0;
}

int WasmWriter::writeSectionContent(raw_ostream &OS,
                                    WasmYAML::LinkingSection &Section) {
  writeStringRef(Section.Name, OS);
  encodeULEB128(Section.Version, OS);

  SubSectionWriter SubSection(OS);

  // SYMBOL_TABLE subsection
  if (Section.SymbolTable.size()) {
    writeUint8(OS, wasm::WASM_SYMBOL_TABLE);

    encodeULEB128(Section.SymbolTable.size(), SubSection.getStream());
#ifndef NDEBUG
    uint32_t SymbolIndex = 0;
#endif
    for (const WasmYAML::SymbolInfo &Info : Section.SymbolTable) {
      assert(Info.Index == SymbolIndex++);
      writeUint8(SubSection.getStream(), Info.Kind);
      encodeULEB128(Info.Flags, SubSection.getStream());
      switch (Info.Kind) {
      case wasm::WASM_SYMBOL_TYPE_FUNCTION:
      case wasm::WASM_SYMBOL_TYPE_GLOBAL:
      case wasm::WASM_SYMBOL_TYPE_EVENT:
        encodeULEB128(Info.ElementIndex, SubSection.getStream());
        if ((Info.Flags & wasm::WASM_SYMBOL_UNDEFINED) == 0)
          writeStringRef(Info.Name, SubSection.getStream());
        break;
      case wasm::WASM_SYMBOL_TYPE_DATA:
        writeStringRef(Info.Name, SubSection.getStream());
        if ((Info.Flags & wasm::WASM_SYMBOL_UNDEFINED) == 0) {
          encodeULEB128(Info.DataRef.Segment, SubSection.getStream());
          encodeULEB128(Info.DataRef.Offset, SubSection.getStream());
          encodeULEB128(Info.DataRef.Size, SubSection.getStream());
        }
        break;
      case wasm::WASM_SYMBOL_TYPE_SECTION:
        encodeULEB128(Info.ElementIndex, SubSection.getStream());
        break;
      default:
        llvm_unreachable("unexpected kind");
      }
    }

    SubSection.done();
  }

  // SEGMENT_NAMES subsection
  if (Section.SegmentInfos.size()) {
    writeUint8(OS, wasm::WASM_SEGMENT_INFO);
    encodeULEB128(Section.SegmentInfos.size(), SubSection.getStream());
    for (const WasmYAML::SegmentInfo &SegmentInfo : Section.SegmentInfos) {
      writeStringRef(SegmentInfo.Name, SubSection.getStream());
      encodeULEB128(SegmentInfo.Alignment, SubSection.getStream());
      encodeULEB128(SegmentInfo.Flags, SubSection.getStream());
    }
    SubSection.done();
  }

  // INIT_FUNCS subsection
  if (Section.InitFunctions.size()) {
    writeUint8(OS, wasm::WASM_INIT_FUNCS);
    encodeULEB128(Section.InitFunctions.size(), SubSection.getStream());
    for (const WasmYAML::InitFunction &Func : Section.InitFunctions) {
      encodeULEB128(Func.Priority, SubSection.getStream());
      encodeULEB128(Func.Symbol, SubSection.getStream());
    }
    SubSection.done();
  }

  // COMDAT_INFO subsection
  if (Section.Comdats.size()) {
    writeUint8(OS, wasm::WASM_COMDAT_INFO);
    encodeULEB128(Section.Comdats.size(), SubSection.getStream());
    for (const auto &C : Section.Comdats) {
      writeStringRef(C.Name, SubSection.getStream());
      encodeULEB128(0, SubSection.getStream()); // flags for future use
      encodeULEB128(C.Entries.size(), SubSection.getStream());
      for (const WasmYAML::ComdatEntry &Entry : C.Entries) {
        writeUint8(SubSection.getStream(), Entry.Kind);
        encodeULEB128(Entry.Index, SubSection.getStream());
      }
    }
    SubSection.done();
  }

  return 0;
}

int WasmWriter::writeSectionContent(raw_ostream &OS,
                                    WasmYAML::NameSection &Section) {
  writeStringRef(Section.Name, OS);
  if (Section.FunctionNames.size()) {
    writeUint8(OS, wasm::WASM_NAMES_FUNCTION);

    SubSectionWriter SubSection(OS);

    encodeULEB128(Section.FunctionNames.size(), SubSection.getStream());
    for (const WasmYAML::NameEntry &NameEntry : Section.FunctionNames) {
      encodeULEB128(NameEntry.Index, SubSection.getStream());
      writeStringRef(NameEntry.Name, SubSection.getStream());
    }

    SubSection.done();
  }
  return 0;
}

int WasmWriter::writeSectionContent(raw_ostream &OS,
                                    WasmYAML::ProducersSection &Section) {
  writeStringRef(Section.Name, OS);
  int Fields = int(!Section.Languages.empty()) + int(!Section.Tools.empty()) +
               int(!Section.SDKs.empty());
  if (Fields == 0)
    return 0;
  encodeULEB128(Fields, OS);
  for (auto &Field : {std::make_pair(StringRef("language"), &Section.Languages),
                      std::make_pair(StringRef("processed-by"), &Section.Tools),
                      std::make_pair(StringRef("sdk"), &Section.SDKs)}) {
    if (Field.second->empty())
      continue;
    writeStringRef(Field.first, OS);
    encodeULEB128(Field.second->size(), OS);
    for (auto &Entry : *Field.second) {
      writeStringRef(Entry.Name, OS);
      writeStringRef(Entry.Version, OS);
    }
  }
  return 0;
}

int WasmWriter::writeSectionContent(raw_ostream &OS,
                                    WasmYAML::CustomSection &Section) {
  if (auto S = dyn_cast<WasmYAML::DylinkSection>(&Section)) {
    if (auto Err = writeSectionContent(OS, *S))
      return Err;
  } else if (auto S = dyn_cast<WasmYAML::NameSection>(&Section)) {
    if (auto Err = writeSectionContent(OS, *S))
      return Err;
  } else if (auto S = dyn_cast<WasmYAML::LinkingSection>(&Section)) {
    if (auto Err = writeSectionContent(OS, *S))
      return Err;
  } else if (auto S = dyn_cast<WasmYAML::ProducersSection>(&Section)) {
    if (auto Err = writeSectionContent(OS, *S))
      return Err;
  } else {
    writeStringRef(Section.Name, OS);
    Section.Payload.writeAsBinary(OS);
  }
  return 0;
}

int WasmWriter::writeSectionContent(raw_ostream &OS,
                                    WasmYAML::TypeSection &Section) {
  encodeULEB128(Section.Signatures.size(), OS);
  uint32_t ExpectedIndex = 0;
  for (const WasmYAML::Signature &Sig : Section.Signatures) {
    if (Sig.Index != ExpectedIndex) {
      errs() << "Unexpected type index: " << Sig.Index << "\n";
      return 1;
    }
    ++ExpectedIndex;
    writeUint8(OS, Sig.Form);
    encodeULEB128(Sig.ParamTypes.size(), OS);
    for (auto ParamType : Sig.ParamTypes)
      writeUint8(OS, ParamType);
    if (Sig.ReturnType == wasm::WASM_TYPE_NORESULT) {
      encodeULEB128(0, OS);
    } else {
      encodeULEB128(1, OS);
      writeUint8(OS, Sig.ReturnType);
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
    writeUint8(OS, Import.Kind);
    switch (Import.Kind) {
    case wasm::WASM_EXTERNAL_FUNCTION:
      encodeULEB128(Import.SigIndex, OS);
      NumImportedFunctions++;
      break;
    case wasm::WASM_EXTERNAL_GLOBAL:
      writeUint8(OS, Import.GlobalImport.Type);
      writeUint8(OS, Import.GlobalImport.Mutable);
      NumImportedGlobals++;
      break;
    case wasm::WASM_EXTERNAL_EVENT:
      writeUint32(OS, Import.EventImport.Attribute);
      writeUint32(OS, Import.EventImport.SigIndex);
      NumImportedGlobals++;
      break;
    case wasm::WASM_EXTERNAL_MEMORY:
      writeLimits(Import.Memory, OS);
      break;
    case wasm::WASM_EXTERNAL_TABLE:
      writeUint8(OS, Import.TableImport.ElemType);
      writeLimits(Import.TableImport.TableLimits, OS);
      break;
    default:
      errs() << "Unknown import type: " << Import.Kind << "\n";
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
    writeUint8(OS, Export.Kind);
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
    writeUint8(OS, Table.ElemType);
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
  uint32_t ExpectedIndex = NumImportedGlobals;
  for (auto &Global : Section.Globals) {
    if (Global.Index != ExpectedIndex) {
      errs() << "Unexpected global index: " << Global.Index << "\n";
      return 1;
    }
    ++ExpectedIndex;
    writeUint8(OS, Global.Type);
    writeUint8(OS, Global.Mutable);
    writeInitExpr(Global.InitExpr, OS);
  }
  return 0;
}

int WasmWriter::writeSectionContent(raw_ostream &OS,
                                    WasmYAML::EventSection &Section) {
  encodeULEB128(Section.Events.size(), OS);
  uint32_t ExpectedIndex = NumImportedEvents;
  for (auto &Event : Section.Events) {
    if (Event.Index != ExpectedIndex) {
      errs() << "Unexpected event index: " << Event.Index << "\n";
      return 1;
    }
    ++ExpectedIndex;
    encodeULEB128(Event.Attribute, OS);
    encodeULEB128(Event.SigIndex, OS);
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
  uint32_t ExpectedIndex = NumImportedFunctions;
  for (auto &Func : Section.Functions) {
    std::string OutString;
    raw_string_ostream StringStream(OutString);
    if (Func.Index != ExpectedIndex) {
      errs() << "Unexpected function index: " << Func.Index << "\n";
      return 1;
    }
    ++ExpectedIndex;

    encodeULEB128(Func.Locals.size(), StringStream);
    for (auto &LocalDecl : Func.Locals) {
      encodeULEB128(LocalDecl.Count, StringStream);
      writeUint8(StringStream, LocalDecl.Type);
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

int WasmWriter::writeRelocSection(raw_ostream &OS, WasmYAML::Section &Sec,
                                  uint32_t SectionIndex) {
  switch (Sec.Type) {
  case wasm::WASM_SEC_CODE:
    writeStringRef("reloc.CODE", OS);
    break;
  case wasm::WASM_SEC_DATA:
    writeStringRef("reloc.DATA", OS);
    break;
  case wasm::WASM_SEC_CUSTOM: {
    auto CustomSection = dyn_cast<WasmYAML::CustomSection>(&Sec);
    if (!CustomSection->Name.startswith(".debug_")) {
      llvm_unreachable("not yet implemented (only for debug sections)");
      return 1;
    }

    writeStringRef(("reloc." + CustomSection->Name).str(), OS);
    break;
  }
  default:
    llvm_unreachable("not yet implemented");
    return 1;
  }

  encodeULEB128(SectionIndex, OS);
  encodeULEB128(Sec.Relocations.size(), OS);

  for (auto Reloc : Sec.Relocations) {
    writeUint8(OS, Reloc.Type);
    encodeULEB128(Reloc.Offset, OS);
    encodeULEB128(Reloc.Index, OS);
    switch (Reloc.Type) {
    case wasm::R_WASM_MEMORY_ADDR_LEB:
    case wasm::R_WASM_MEMORY_ADDR_SLEB:
    case wasm::R_WASM_MEMORY_ADDR_I32:
    case wasm::R_WASM_FUNCTION_OFFSET_I32:
    case wasm::R_WASM_SECTION_OFFSET_I32:
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
  llvm::object::WasmSectionOrderChecker Checker;
  for (const std::unique_ptr<WasmYAML::Section> &Sec : Obj.Sections) {
    StringRef SecName = "";
    if (auto S = dyn_cast<WasmYAML::CustomSection>(Sec.get()))
      SecName = S->Name;
    if (!Checker.isValidSectionOrder(Sec->Type, SecName)) {
      errs() << "Out of order section type: " << Sec->Type << "\n";
      return 1;
    }
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
    } else if (auto S = dyn_cast<WasmYAML::EventSection>(Sec.get())) {
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
  uint32_t SectionIndex = 0;
  for (const std::unique_ptr<WasmYAML::Section> &Sec : Obj.Sections) {
    if (Sec->Relocations.empty()) {
      SectionIndex++;
      continue;
    }

    writeUint8(OS, wasm::WASM_SEC_CUSTOM);
    std::string OutString;
    raw_string_ostream StringStream(OutString);
    writeRelocSection(StringStream, *Sec, SectionIndex++);
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
