//===- ObjectFileTransformer.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/DebugInfo/GSYM/GsymCreator.h"
#include "llvm/DebugInfo/GSYM/ObjectFileTransformer.h"
#include "llvm/DebugInfo/GSYM/OutputAggregator.h"

using namespace llvm;
using namespace gsym;

constexpr uint32_t NT_GNU_BUILD_ID_TAG = 0x03;

static std::vector<uint8_t> getUUID(const object::ObjectFile &Obj) {
  // Extract the UUID from the object file
  std::vector<uint8_t> UUID;
  if (auto *MachO = dyn_cast<object::MachOObjectFile>(&Obj)) {
    const ArrayRef<uint8_t> MachUUID = MachO->getUuid();
    if (!MachUUID.empty())
      UUID.assign(MachUUID.data(), MachUUID.data() + MachUUID.size());
  } else if (isa<object::ELFObjectFileBase>(&Obj)) {
    const StringRef GNUBuildID(".note.gnu.build-id");
    for (const object::SectionRef &Sect : Obj.sections()) {
      Expected<StringRef> SectNameOrErr = Sect.getName();
      if (!SectNameOrErr) {
        consumeError(SectNameOrErr.takeError());
        continue;
      }
      StringRef SectName(*SectNameOrErr);
      if (SectName != GNUBuildID)
        continue;
      StringRef BuildIDData;
      Expected<StringRef> E = Sect.getContents();
      if (E)
        BuildIDData = *E;
      else {
        consumeError(E.takeError());
        continue;
      }
      DataExtractor Decoder(BuildIDData, Obj.makeTriple().isLittleEndian());
      uint64_t Offset = 0;
      const uint32_t NameSize = Decoder.getU32(&Offset);
      const uint32_t PayloadSize = Decoder.getU32(&Offset);
      const uint32_t PayloadType = Decoder.getU32(&Offset);
      StringRef Name(Decoder.getFixedLengthString(&Offset, NameSize));
      if (Name == "GNU" && PayloadType == NT_GNU_BUILD_ID_TAG) {
        Offset = alignTo(Offset, 4);
        StringRef UUIDBytes(Decoder.getBytes(&Offset, PayloadSize));
        if (!UUIDBytes.empty()) {
          auto Ptr = reinterpret_cast<const uint8_t *>(UUIDBytes.data());
          UUID.assign(Ptr, Ptr + UUIDBytes.size());
        }
      }
    }
  }
  return UUID;
}

/// Create function information entries for Mach-O symbol stubs.
///
/// Symbol stubs are small chunks of code, all with the same fixed size, that
/// resolves a function pointer on first call and then jumps to the resolved
/// function on subsequent calls for functions that live in another shared
/// library. The stubs are stored in sections whose type is S_SYMBOL_STUBS and
/// the size of a single stub is stored in the "reserved2" field of the section
/// header. These stubs have no entries in the symbol table of their own, so
/// they end up being attributed to whatever function precedes them unless they
/// are synthesized here.
///
/// The name of the function a stub jumps to is found using the indirect symbol
/// table from the LC_DYSYMTAB load command. The "reserved1" field of the
/// section header contains the index of the indirect symbol table entry that
/// describes the first stub in the section, and each subsequent stub is
/// described by the entry that follows. Each indirect symbol table entry is an
/// index into the symbol table where the matching undefined (N_UNDF) symbol
/// supplies the name to use for the stub. Each name gets a "symbol stub for: "
/// prefix prepended to it so that symbolication makes it clear that the address
/// is the stub for a function and not the function itself.
///
/// \returns The number of function infos that were added to \a Gsym.
static uint64_t addMachOSymbolStubs(const object::MachOObjectFile &MachO,
                                    OutputAggregator &Out, GsymCreator &Gsym) {
  const MachO::dysymtab_command Dysymtab = MachO.getDysymtabLoadCommand();
  if (Dysymtab.nindirectsyms == 0)
    return 0;
  const uint32_t NumSyms = MachO.getSymtabLoadCommand().nsyms;
  const bool Is64Bit = MachO.is64Bit();

  size_t NumBefore = Gsym.getNumFunctionInfos();
  for (const object::SectionRef &Sect : MachO.sections()) {
    const object::DataRefImpl SectDRI = Sect.getRawDataRefImpl();
    uint32_t SectFlags, IndirectSymIdxStart, StubByteSize;
    if (Is64Bit) {
      const MachO::section_64 S = MachO.getSection64(SectDRI);
      SectFlags = S.flags;
      IndirectSymIdxStart = S.reserved1;
      StubByteSize = S.reserved2;
    } else {
      const MachO::section S = MachO.getSection(SectDRI);
      SectFlags = S.flags;
      IndirectSymIdxStart = S.reserved1;
      StubByteSize = S.reserved2;
    }
    if ((SectFlags & MachO::SECTION_TYPE) != MachO::S_SYMBOL_STUBS)
      continue;
    if (StubByteSize == 0)
      continue;

    const uint64_t SectAddr = Sect.getAddress();
    const uint64_t NumStubs = Sect.getSize() / StubByteSize;
    for (uint64_t StubIdx = 0; StubIdx < NumStubs; ++StubIdx) {
      const uint64_t StubAddr = SectAddr + StubIdx * StubByteSize;
      if (!Gsym.IsValidTextAddress(StubAddr))
        continue;
      const uint64_t IndirectSymIdx = IndirectSymIdxStart + StubIdx;
      if (IndirectSymIdx >= Dysymtab.nindirectsyms)
        continue;
      const uint32_t SymIdx =
          MachO.getIndirectSymbolTableEntry(Dysymtab, IndirectSymIdx);
      // Entries that are absolute or local don't refer to a symbol table entry.
      if (SymIdx & (MachO::INDIRECT_SYMBOL_ABS | MachO::INDIRECT_SYMBOL_LOCAL))
        continue;
      if (SymIdx >= NumSyms)
        continue;
      const object::symbol_iterator SymIt = MachO.getSymbolByIndex(SymIdx);
      const object::DataRefImpl SymDRI = SymIt->getRawDataRefImpl();
      const uint8_t NType = Is64Bit ? MachO.getSymbol64TableEntry(SymDRI).n_type
                                    : MachO.getSymbolTableEntry(SymDRI).n_type;
      // Only undefined symbols name a stub, any other symbol type means the
      // function itself is in this file and already has a symbol table entry.
      if ((NType & MachO::N_TYPE) != MachO::N_UNDF)
        continue;
      Expected<StringRef> Name = SymIt->getName();
      if (!Name) {
        if (Out.GetOS())
          logAllUnhandledErrors(Name.takeError(), *Out.GetOS(),
                                "ObjectFileTransformer: ");
        else
          consumeError(Name.takeError());
        continue;
      }
      // Remove the leading '_' character in any symbol names if there is one
      // for mach-o files.
      Name->consume_front("_");
      if (Name->empty())
        continue;
      // Append a "symbol stub for: " prefix so it is clear when symbolicating
      // that the address is the stub for the function and not the function
      // itself. The string must be copied into the string table since it is
      // created here and, unlike the symbol names, has no backing storage in
      // the object file.
      constexpr bool Copy = true;
      const std::string StubName = "symbol stub for: " + Name->str();
      Gsym.addFunctionInfo(FunctionInfo(StubAddr, StubByteSize,
                                        Gsym.insertString(StubName, Copy)));
    }
  }
  return Gsym.getNumFunctionInfos() - NumBefore;
}

static Expected<size_t>
loadSymbols(const object::ObjectFile &Obj,
            object::ObjectFile::symbol_iterator_range Symbols,
            OutputAggregator &Out, GsymCreator &Gsym, bool CopyStrings) {
  using namespace llvm::object;

  const bool IsMachO = isa<MachOObjectFile>(&Obj);
  const bool IsELF = isa<ELFObjectFileBase>(&Obj);
  const size_t NumBefore = Gsym.getNumFunctionInfos();

  for (const object::SymbolRef &Sym : Symbols) {
    Expected<SymbolRef::Type> SymType = Sym.getType();
    if (!SymType) {
      consumeError(SymType.takeError());
      continue;
    }
    Expected<uint64_t> AddrOrErr = Sym.getValue();
    if (!AddrOrErr)
      // TODO: Test this error.
      return AddrOrErr.takeError();

    if (SymType.get() != SymbolRef::Type::ST_Function ||
        !Gsym.IsValidTextAddress(*AddrOrErr))
      continue;
    // Function size for MachO files will be 0
    const uint64_t size = IsELF ? ELFSymbolRef(Sym).getSize() : 0;
    Expected<StringRef> Name = Sym.getName();
    if (!Name) {
      if (Out.GetOS())
        logAllUnhandledErrors(Name.takeError(), *Out.GetOS(),
                              "ObjectFileTransformer: ");
      else
        consumeError(Name.takeError());
      continue;
    }
    // Remove the leading '_' character in any symbol names if there is one
    // for mach-o files.
    if (IsMachO)
      Name->consume_front("_");
    Gsym.addFunctionInfo(
        FunctionInfo(*AddrOrErr, size, Gsym.insertString(*Name, CopyStrings)));
  }
  return Gsym.getNumFunctionInfos() - NumBefore;
}

// Add function symbols from the .gnu_debugdata section (MiniDebugInfo), which
// contains an xz-compressed ELF file with a .symtab. Errors reading the section
// are reported as warnings.
static llvm::Error loadGnuDebugDataSymbols(const object::ELFObjectFileBase &Obj,
                                           OutputAggregator &Out,
                                           GsymCreator &Gsym) {
  using namespace llvm::object;

  std::optional<SectionRef> DebugDataSect;
  for (const SectionRef &Sect : Obj.sections()) {
    Expected<StringRef> SectNameOrErr = Sect.getName();
    if (!SectNameOrErr) {
      consumeError(SectNameOrErr.takeError());
      continue;
    }
    if (*SectNameOrErr == ".gnu_debugdata") {
      DebugDataSect = Sect;
      break;
    }
  }
  if (!DebugDataSect)
    return Error::success();

  // MiniDebugInfo leaves out symbols that are in .dynsym, so add those too.
  Expected<size_t> DynFunctionsAdded = loadSymbols(
      Obj, Obj.getDynamicSymbolIterators(), Out, Gsym, /*CopyStrings=*/false);
  if (!DynFunctionsAdded)
    return DynFunctionsAdded.takeError();
  if (Out.GetOS())
    *Out.GetOS() << "Loaded " << *DynFunctionsAdded
                 << " functions from dynamic symbol table.\n";

  auto Warn = [&](const Twine &Msg) {
    Out.Report(
        "Failed to load the .gnu_debugdata section", [&](raw_ostream &OS) {
          OS << "warning: unable to read the .gnu_debugdata section: " << Msg
             << "\n";
        });
    return Error::success();
  };

  if (!compression::xz::isAvailable())
    return Warn("missing LZMA support (LLVM_ENABLE_LZMA)");
  Expected<StringRef> Contents = DebugDataSect->getContents();
  if (!Contents)
    return Warn(toString(Contents.takeError()));
  SmallVector<uint8_t, 0> Decompressed;
  if (Error E = compression::xz::decompress(arrayRefFromStringRef(*Contents),
                                            Decompressed))
    return Warn("failed to decompress: " + toString(std::move(E)));
  Expected<std::unique_ptr<ObjectFile>> DebugObj =
      ObjectFile::createELFObjectFile(
          MemoryBufferRef(toStringRef(Decompressed), Obj.getFileName()));
  if (!DebugObj)
    return Warn("failed to parse the embedded ELF object: " +
                toString(DebugObj.takeError()));

  // Decompressed is freed when we return, so the names need to be copied.
  Expected<size_t> DebugFunctionsAdded =
      loadSymbols(**DebugObj, (*DebugObj)->symbols(), Out, Gsym,
                  /*CopyStrings=*/true);
  if (!DebugFunctionsAdded)
    return Warn(toString(DebugFunctionsAdded.takeError()));
  if (Out.GetOS())
    *Out.GetOS() << "Loaded " << *DebugFunctionsAdded
                 << " functions from .gnu_debugdata symbol table.\n";
  return Error::success();
}

llvm::Error ObjectFileTransformer::convert(const object::ObjectFile &Obj,
                                           OutputAggregator &Out,
                                           GsymCreator &Gsym) {
  using namespace llvm::object;

  // Read build ID.
  Gsym.setUUID(getUUID(Obj));

  // Parse the symbol table.
  Expected<size_t> FunctionsAddedCount =
      loadSymbols(Obj, Obj.symbols(), Out, Gsym, /*CopyStrings=*/false);
  if (!FunctionsAddedCount)
    return FunctionsAddedCount.takeError();
  if (Out.GetOS())
    *Out.GetOS() << "Loaded " << *FunctionsAddedCount
                 << " functions from symbol table.\n";

  // Mach-O symbol stubs have no symbol table entries of their own, so
  // synthesize function infos for them using the indirect symbol table.
  if (const auto *MachO = dyn_cast<MachOObjectFile>(&Obj)) {
    const uint64_t StubsAddedCount = addMachOSymbolStubs(*MachO, Out, Gsym);
    if (Out.GetOS())
      *Out.GetOS() << "Loaded " << StubsAddedCount
                   << " functions from symbol stubs.\n";
  }

  if (const auto *ELFObj = dyn_cast<ELFObjectFileBase>(&Obj))
    return loadGnuDebugDataSymbols(*ELFObj, Out, Gsym);
  return Error::success();
}
