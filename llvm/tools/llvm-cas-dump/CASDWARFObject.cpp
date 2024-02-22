//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CASDWARFObject.h"
#include "llvm/CASObjectFormats/Encoding.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAbbrev.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/Object/MachO.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::mccasformats::v1;

namespace {
/// Parse the MachO header to extract details such as endianness.
/// Unfortunately object::MachOObjectfile() doesn't support parsing
/// incomplete files.
struct MachOHeaderParser {
  bool Is64Bit = true;
  bool IsLittleEndian = true;

  /// Stolen from MachOObjectfile.
  template <typename T>
  Expected<T> getStructOrErr(StringRef Data, const char *P) {
    // Don't read before the beginning or past the end of the file
    if (P < Data.begin() || P + sizeof(T) > Data.end())
      return make_error<llvm::object::GenericBinaryError>(
          "Structure read out-of-range");

    T Cmd;
    memcpy(&Cmd, P, sizeof(T));
    if (IsLittleEndian != sys::IsLittleEndianHost)
      MachO::swapStruct(Cmd);
    return Cmd;
  }

  /// Parse an mc::header.
  Error parse(StringRef Data) {
    // MachO 64-bit header.
    const char *P = Data.data();
    auto Header64 = getStructOrErr<MachO::mach_header_64>(Data, P);
    P += sizeof(MachO::mach_header_64);
    if (!Header64)
      return Header64.takeError();
    if (Header64->magic == MachO::MH_MAGIC_64) {
      Is64Bit = true;
      IsLittleEndian = true;
    } else {
      return make_error<object::GenericBinaryError>("Unsupported MachO format");
    }
    return Error::success();
  }
};
} // namespace

Error CASDWARFObject::discoverDwarfSections(ObjectRef CASObj) {
  if (CASObj == Schema.getRootNodeTypeID())
    return Error::success();
  Expected<MCObjectProxy> MCObj = Schema.get(CASObj);
  if (!MCObj)
    return MCObj.takeError();
  return discoverDwarfSections(*MCObj);
}

Error CASDWARFObject::discoverDwarfSections(MCObjectProxy MCObj) {
  StringRef Data = MCObj.getData();
  if (auto MCAssRef = MCAssemblerRef::Cast(MCObj)) {
    StringRef Remaining = MCAssRef->getData();
    uint32_t NormalizedTripleSize;
    if (auto E = casobjectformats::encoding::consumeVBR8(Remaining,
                                                         NormalizedTripleSize))
      return E;
    auto TripleStr = Remaining.take_front(NormalizedTripleSize);
    Triple Target(TripleStr);
    this->Target = Target;
  }
  if (HeaderRef::Cast(MCObj)) {
    MachOHeaderParser P;
    if (Error Err = P.parse(MCObj.getData()))
      return Err;
    Is64Bit = P.Is64Bit;
    IsLittleEndian = P.IsLittleEndian;
  } else if (auto LineSectionRef = DebugLineSectionRef::Cast(MCObj)) {
    raw_svector_ostream OS(DebugLineSection);
    MCCASReader Reader(OS, Target, MCObj.getSchema());
    auto Written = LineSectionRef->materialize(Reader);
    if (!Written)
      return Written.takeError();
  }
  if (DebugStrRef::Cast(MCObj)) {
    MapOfStringOffsets.try_emplace(MCObj.getRef(), DebugStringSection.size());
    DebugStringSection.append(Data.begin(), Data.end());
    DebugStringSection.push_back(0);
  }
  if (DebugAbbrevSectionRef::Cast(MCObj) || GroupRef::Cast(MCObj) ||
      SymbolTableRef::Cast(MCObj) || SectionRef::Cast(MCObj) ||
      DebugLineSectionRef::Cast(MCObj) || AtomRef::Cast(MCObj)) {
    auto Refs = MCObjectProxy::decodeReferences(MCObj, Data);
    if (!Refs)
      return Refs.takeError();
    for (auto Ref : *Refs) {
      if (Error E = discoverDwarfSections(Ref))
        return E;
    }
    return Error::success();
  }
  return MCObj.forEachReference(
      [this](ObjectRef CASObj) { return discoverDwarfSections(CASObj); });
}

Error CASDWARFObject::dump(raw_ostream &OS, int Indent, DWARFContext &DWARFCtx,
                           MCObjectProxy MCObj, bool Verbose) {
  OS.indent(Indent);
  DIDumpOptions DumpOpts;
  DumpOpts.ShowChildren = true;
  DumpOpts.Verbose = Verbose;
  StringRef Data = MCObj.getData();
  if (Data.empty())
    return Error::success();
  if (DebugStrRef::Cast(MCObj)) {
    // Dump __debug_str data.
    assert(Data.data()[Data.size()] == 0);
    DataExtractor StrData(StringRef(Data.data(), Data.size() + 1),
                          isLittleEndian(), 0);
    // This is almost identical with the DumpStrSection lambda in
    // DWARFContext.cpp
    uint64_t Offset = 0;
    uint64_t StrOffset = 0;
    Error Err = Error::success();
    while (StrData.isValidOffset(Offset)) {
      const char *CStr = StrData.getCStr(&Offset, &Err);
      if (Err)
        return Err;
      OS << format("0x%8.8" PRIx64 ": \"", StrOffset);
      OS.write_escaped(CStr);
      OS << "\"\n";
      StrOffset = Offset;
    }
  } else if (DebugLineSectionRef::Cast(MCObj)) {
    // Dump __debug_line data.
    uint64_t Address = 0;
    DWARFDataExtractor LineData(*this, {toStringRef(DebugLineSection), Address},
                                isLittleEndian(), 0);
    DWARFDebugLine::SectionParser Parser(LineData, DWARFCtx,
                                         DWARFCtx.normal_units());
    while (!Parser.done()) {
      OS << "debug_line[" << format("0x%8.8" PRIx64, Parser.getOffset())
         << "]\n";
      Parser.parseNext(DumpOpts.WarningHandler, DumpOpts.WarningHandler, &OS,
                       DumpOpts.Verbose);
    }
  }
  return Error::success();
}
