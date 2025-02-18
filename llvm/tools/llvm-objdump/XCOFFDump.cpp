//===-- XCOFFDump.cpp - XCOFF-specific dumper -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the XCOFF-specific dumper for llvm-objdump.
///
//===----------------------------------------------------------------------===//

#include "XCOFFDump.h"

#include "llvm-objdump.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::XCOFF;
using namespace llvm::support;

namespace {
class XCOFFDumper : public objdump::Dumper {
  enum PrintStyle { Hex, Number };
  const XCOFFObjectFile &Obj;
  unsigned Width;

public:
  XCOFFDumper(const object::XCOFFObjectFile &O) : Dumper(O), Obj(O) {}

private:
  void printPrivateHeaders() override;
  void printFileHeader();
  void printAuxiliaryHeader();
  void printLoaderSectionHeader();
  void printAuxiliaryHeader(const XCOFFAuxiliaryHeader32 *AuxHeader);
  void printAuxiliaryHeader(const XCOFFAuxiliaryHeader64 *AuxHeader);
  template <typename AuxHeaderMemberType, typename XCOFFAuxiliaryHeader>
  void printAuxMemberHelper(PrintStyle Style, const char *MemberName,
                            const AuxHeaderMemberType &Member,
                            const XCOFFAuxiliaryHeader *AuxHeader,
                            uint16_t AuxSize, uint16_t &PartialFieldOffset,
                            const char *&PartialFieldName);
  template <typename XCOFFAuxiliaryHeader>
  void checkAndPrintAuxHeaderParseError(const char *PartialFieldName,
                                        uint16_t PartialFieldOffset,
                                        uint16_t AuxSize,
                                        XCOFFAuxiliaryHeader &AuxHeader);

  void printBinary(StringRef Name, ArrayRef<uint8_t> Data);
  void printHex(StringRef Name, uint64_t Value);
  void printNumber(StringRef Name, uint64_t Value);
  FormattedString formatName(StringRef Name);
  void printStrHex(StringRef Name, StringRef Str, uint64_t Value);
};

void XCOFFDumper::printPrivateHeaders() {
  printFileHeader();
  printAuxiliaryHeader();
  printLoaderSectionHeader();
}

FormattedString XCOFFDumper::formatName(StringRef Name) {
  return FormattedString(Name, Width, FormattedString::JustifyLeft);
}

void XCOFFDumper::printHex(StringRef Name, uint64_t Value) {
  outs() << formatName(Name) << format_hex(Value, 0) << "\n";
}

void XCOFFDumper::printNumber(StringRef Name, uint64_t Value) {
  outs() << formatName(Name) << format_decimal(Value, 0) << "\n";
}

void XCOFFDumper::printStrHex(StringRef Name, StringRef Str, uint64_t Value) {
  outs() << formatName(Name) << Str << " (" << format_decimal(Value, 0)
         << ")\n";
}

void XCOFFDumper::printBinary(StringRef Name, ArrayRef<uint8_t> Data) {
  unsigned OrgWidth = Width;
  Width = 0;
  outs() << formatName(Name) << " (" << format_bytes(Data) << ")\n";
  Width = OrgWidth;
}

void XCOFFDumper::printAuxiliaryHeader() {
  Width = 36;
  if (Obj.is64Bit())
    printAuxiliaryHeader(Obj.auxiliaryHeader64());
  else
    printAuxiliaryHeader(Obj.auxiliaryHeader32());
}

template <typename AuxHeaderMemberType, typename XCOFFAuxiliaryHeader>
void XCOFFDumper::printAuxMemberHelper(PrintStyle Style, const char *MemberName,
                                       const AuxHeaderMemberType &Member,
                                       const XCOFFAuxiliaryHeader *AuxHeader,
                                       uint16_t AuxSize,
                                       uint16_t &PartialFieldOffset,
                                       const char *&PartialFieldName) {
  ptrdiff_t Offset = reinterpret_cast<const char *>(&Member) -
                     reinterpret_cast<const char *>(AuxHeader);
  if (Offset + sizeof(Member) <= AuxSize) {
    if (Style == Hex)
      printHex(MemberName, Member);
    else
      printNumber(MemberName, Member);
  } else if (Offset < AuxSize) {
    PartialFieldOffset = Offset;
    PartialFieldName = MemberName;
  }
}

template <typename XCOFFAuxiliaryHeader>
void XCOFFDumper::checkAndPrintAuxHeaderParseError(
    const char *PartialFieldName, uint16_t PartialFieldOffset, uint16_t AuxSize,
    XCOFFAuxiliaryHeader &AuxHeader) {
  if (PartialFieldOffset < AuxSize) {
    std::string Buf;
    raw_string_ostream OS(Buf);
    OS.flush();
    OS << FormattedString("Raw data", 0, FormattedString::JustifyLeft) << " ("
       << format_bytes(
              ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(&AuxHeader) +
                                    PartialFieldOffset,
                                AuxSize - PartialFieldOffset))
       << ")\n";
    reportUniqueWarning(Twine("only partial field for ") + PartialFieldName +
                        " at offset (" + Twine(PartialFieldOffset) + ")\n" +
                        OS.str());
  } else if (sizeof(AuxHeader) < AuxSize) {
    printBinary(
        "Extra raw data",
        ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(&AuxHeader) +
                              sizeof(AuxHeader),
                          AuxSize - sizeof(AuxHeader)));
  }
}

void XCOFFDumper::printAuxiliaryHeader(
    const XCOFFAuxiliaryHeader32 *AuxHeader) {
  if (AuxHeader == nullptr)
    return;
  outs() << "\n---Auxiliary Header:\n";
  uint16_t AuxSize = Obj.getOptionalHeaderSize();
  uint16_t PartialFieldOffset = AuxSize;
  const char *PartialFieldName = nullptr;

  auto PrintAuxMember = [&](PrintStyle Style, const char *MemberName,
                            auto &Member) {
    printAuxMemberHelper(Style, MemberName, Member, AuxHeader, AuxSize,
                         PartialFieldOffset, PartialFieldName);
  };

  PrintAuxMember(Hex, "Magic:", AuxHeader->AuxMagic);
  PrintAuxMember(Hex, "Version:", AuxHeader->Version);
  PrintAuxMember(Hex, "Size of .text section:", AuxHeader->TextSize);
  PrintAuxMember(Hex, "Size of .data section:", AuxHeader->InitDataSize);
  PrintAuxMember(Hex, "Size of .bss section:", AuxHeader->BssDataSize);
  PrintAuxMember(Hex, "Entry point address:", AuxHeader->EntryPointAddr);
  PrintAuxMember(Hex, ".text section start address:", AuxHeader->TextStartAddr);
  PrintAuxMember(Hex, ".data section start address:", AuxHeader->DataStartAddr);
  PrintAuxMember(Hex, "TOC anchor address:", AuxHeader->TOCAnchorAddr);
  PrintAuxMember(
      Number, "Section number of entryPoint:", AuxHeader->SecNumOfEntryPoint);
  PrintAuxMember(Number, "Section number of .text:", AuxHeader->SecNumOfText);
  PrintAuxMember(Number, "Section number of .data:", AuxHeader->SecNumOfData);
  PrintAuxMember(Number, "Section number of TOC:", AuxHeader->SecNumOfTOC);
  PrintAuxMember(Number,
                 "Section number of loader data:", AuxHeader->SecNumOfLoader);
  PrintAuxMember(Number, "Section number of .bss:", AuxHeader->SecNumOfBSS);
  PrintAuxMember(Hex, "Maxium alignment of .text:", AuxHeader->MaxAlignOfText);
  PrintAuxMember(Hex, "Maxium alignment of .data:", AuxHeader->MaxAlignOfData);
  PrintAuxMember(Hex, "Module type:", AuxHeader->ModuleType);
  PrintAuxMember(Hex, "CPU type of objects:", AuxHeader->CpuFlag);
  PrintAuxMember(Hex, "Maximum stack size:", AuxHeader->MaxStackSize);
  PrintAuxMember(Hex, "Maximum data size:", AuxHeader->MaxDataSize);
  PrintAuxMember(Hex, "Reserved for debugger:", AuxHeader->ReservedForDebugger);
  PrintAuxMember(Hex, "Text page size:", AuxHeader->TextPageSize);
  PrintAuxMember(Hex, "Data page size:", AuxHeader->DataPageSize);
  PrintAuxMember(Hex, "Stack page size:", AuxHeader->StackPageSize);
  if (offsetof(XCOFFAuxiliaryHeader32, FlagAndTDataAlignment) +
          sizeof(XCOFFAuxiliaryHeader32::FlagAndTDataAlignment) <=
      AuxSize) {
    printHex("Flag:", AuxHeader->getFlag());
    printHex("Alignment of thread-local storage:",
             AuxHeader->getTDataAlignment());
  }

  PrintAuxMember(Number,
                 "Section number for .tdata:", AuxHeader->SecNumOfTData);
  PrintAuxMember(Number, "Section number for .tbss:", AuxHeader->SecNumOfTBSS);

  checkAndPrintAuxHeaderParseError(PartialFieldName, PartialFieldOffset,
                                   AuxSize, *AuxHeader);
}

void XCOFFDumper::printAuxiliaryHeader(
    const XCOFFAuxiliaryHeader64 *AuxHeader) {
  if (AuxHeader == nullptr)
    return;
  uint16_t AuxSize = Obj.getOptionalHeaderSize();
  outs() << "\n---Auxiliary Header:\n";
  uint16_t PartialFieldOffset = AuxSize;
  const char *PartialFieldName = nullptr;

  auto PrintAuxMember = [&](PrintStyle Style, const char *MemberName,
                            auto &Member) {
    printAuxMemberHelper(Style, MemberName, Member, AuxHeader, AuxSize,
                         PartialFieldOffset, PartialFieldName);
  };

  PrintAuxMember(Hex, "Magic:", AuxHeader->AuxMagic);
  PrintAuxMember(Hex, "Version:", AuxHeader->Version);
  PrintAuxMember(Hex, "Reserved for debugger:", AuxHeader->ReservedForDebugger);
  PrintAuxMember(Hex, ".text section start address:", AuxHeader->TextStartAddr);
  PrintAuxMember(Hex, ".data section start address:", AuxHeader->DataStartAddr);
  PrintAuxMember(Hex, "TOC anchor address:", AuxHeader->TOCAnchorAddr);
  PrintAuxMember(
      Number, "Section number of entryPoint:", AuxHeader->SecNumOfEntryPoint);
  PrintAuxMember(Number, "Section number of .text:", AuxHeader->SecNumOfText);
  PrintAuxMember(Number, "Section number of .data:", AuxHeader->SecNumOfData);
  PrintAuxMember(Number, "Section number of TOC:", AuxHeader->SecNumOfTOC);
  PrintAuxMember(Number,
                 "Section number of loader data:", AuxHeader->SecNumOfLoader);
  PrintAuxMember(Number, "Section number of .bss:", AuxHeader->SecNumOfBSS);
  PrintAuxMember(Hex, "Maxium alignment of .text:", AuxHeader->MaxAlignOfText);
  PrintAuxMember(Hex, "Maxium alignment of .data:", AuxHeader->MaxAlignOfData);
  PrintAuxMember(Hex, "Module type:", AuxHeader->ModuleType);
  PrintAuxMember(Hex, "CPU type of objects:", AuxHeader->CpuFlag);
  PrintAuxMember(Hex, "Text page size:", AuxHeader->TextPageSize);
  PrintAuxMember(Hex, "Data page size:", AuxHeader->DataPageSize);
  PrintAuxMember(Hex, "Stack page size:", AuxHeader->StackPageSize);
  if (offsetof(XCOFFAuxiliaryHeader64, FlagAndTDataAlignment) +
          sizeof(XCOFFAuxiliaryHeader64::FlagAndTDataAlignment) <=
      AuxSize) {
    printHex("Flag:", AuxHeader->getFlag());
    printHex("Alignment of thread-local storage:",
             AuxHeader->getTDataAlignment());
  }
  PrintAuxMember(Hex, "Size of .text section:", AuxHeader->TextSize);
  PrintAuxMember(Hex, "Size of .data section:", AuxHeader->InitDataSize);
  PrintAuxMember(Hex, "Size of .bss section:", AuxHeader->BssDataSize);
  PrintAuxMember(Hex, "Entry point address:", AuxHeader->EntryPointAddr);
  PrintAuxMember(Hex, "Maximum stack size:", AuxHeader->MaxStackSize);
  PrintAuxMember(Hex, "Maximum data size:", AuxHeader->MaxDataSize);
  PrintAuxMember(Number,
                 "Section number for .tdata:", AuxHeader->SecNumOfTData);
  PrintAuxMember(Number, "Section number for .tbss:", AuxHeader->SecNumOfTBSS);
  PrintAuxMember(Hex, "Additional flags 64-bit XCOFF:", AuxHeader->XCOFF64Flag);

  checkAndPrintAuxHeaderParseError(PartialFieldName, PartialFieldOffset,
                                   AuxSize, *AuxHeader);
}

void XCOFFDumper::printLoaderSectionHeader() {
  Expected<uintptr_t> LoaderSectionAddrOrError =
      Obj.getSectionFileOffsetToRawData(XCOFF::STYP_LOADER);
  if (!LoaderSectionAddrOrError) {
    reportUniqueWarning(LoaderSectionAddrOrError.takeError());
    return;
  }
  uintptr_t LoaderSectionAddr = LoaderSectionAddrOrError.get();

  if (LoaderSectionAddr == 0)
    return;

  auto PrintLoadSecHeaderCommon = [&](const auto *LDHeader) {
    printNumber("Version:", LDHeader->Version);
    printNumber("NumberOfSymbolEntries:", LDHeader->NumberOfSymTabEnt);
    printNumber("NumberOfRelocationEntries:", LDHeader->NumberOfRelTabEnt);
    printNumber("LengthOfImportFileIDStringTable:",
                LDHeader->LengthOfImpidStrTbl);
    printNumber("NumberOfImportFileIDs:", LDHeader->NumberOfImpid);
    printHex("OffsetToImportFileIDs:", LDHeader->OffsetToImpid);
    printNumber("LengthOfStringTable:", LDHeader->LengthOfStrTbl);
    printHex("OffsetToStringTable:", LDHeader->OffsetToStrTbl);
  };

  Width = 35;
  outs() << "\n---Loader Section Header:\n";
  if (Obj.is64Bit()) {
    const LoaderSectionHeader64 *LoaderSec64 =
        reinterpret_cast<const LoaderSectionHeader64 *>(LoaderSectionAddr);
    PrintLoadSecHeaderCommon(LoaderSec64);
    printHex("OffsetToSymbolTable", LoaderSec64->OffsetToSymTbl);
    printHex("OffsetToRelocationEntries", LoaderSec64->OffsetToRelEnt);
  } else {
    const LoaderSectionHeader32 *LoaderSec32 =
        reinterpret_cast<const LoaderSectionHeader32 *>(LoaderSectionAddr);
    PrintLoadSecHeaderCommon(LoaderSec32);
  }
}

void XCOFFDumper::printFileHeader() {
  Width = 20;
  outs() << "\n---File Header:\n";
  printHex("Magic:", Obj.getMagic());
  printNumber("NumberOfSections:", Obj.getNumberOfSections());

  int32_t Timestamp = Obj.getTimeStamp();
  if (Timestamp > 0) {
    // This handling of the timestamp assumes that the host system's time_t is
    // compatible with AIX time_t. If a platform is not compatible, the lit
    // tests will let us know.
    time_t TimeDate = Timestamp;

    char FormattedTime[20] = {};

    size_t BytesFormatted = std::strftime(FormattedTime, sizeof(FormattedTime),
                                          "%F %T", std::gmtime(&TimeDate));
    assert(BytesFormatted && "The size of the buffer FormattedTime is less "
                             "than the size of the date/time string.");
    (void)BytesFormatted;
    printStrHex("Timestamp:", FormattedTime, Timestamp);
  } else {
    // Negative timestamp values are reserved for future use.
    printStrHex("Timestamp:", Timestamp == 0 ? "None" : "Reserved Value",
                Timestamp);
  }

  // The number of symbol table entries is an unsigned value in 64-bit objects
  // and a signed value (with negative values being 'reserved') in 32-bit
  // objects.
  if (Obj.is64Bit()) {
    printHex("SymbolTableOffset:", Obj.getSymbolTableOffset64());
    printNumber("SymbolTableEntries:", Obj.getNumberOfSymbolTableEntries64());
  } else {
    printHex("SymbolTableOffset:", Obj.getSymbolTableOffset32());
    int32_t SymTabEntries = Obj.getRawNumberOfSymbolTableEntries32();
    if (SymTabEntries >= 0)
      printNumber("SymbolTableEntries:", SymTabEntries);
    else
      printStrHex("SymbolTableEntries:", "Reserved Value", SymTabEntries);
  }

  printHex("OptionalHeaderSize:", Obj.getOptionalHeaderSize());
  printHex("Flags:", Obj.getFlags());
}

} // namespace

std::unique_ptr<objdump::Dumper>
objdump::createXCOFFDumper(const object::XCOFFObjectFile &Obj) {
  return std::make_unique<XCOFFDumper>(Obj);
}

Error objdump::getXCOFFRelocationValueString(const XCOFFObjectFile &Obj,
                                             const RelocationRef &Rel,
                                             bool SymbolDescription,
                                             SmallVectorImpl<char> &Result) {
  symbol_iterator SymI = Rel.getSymbol();
  if (SymI == Obj.symbol_end())
    return make_error<GenericBinaryError>(
        "invalid symbol reference in relocation entry",
        object_error::parse_failed);

  Expected<StringRef> SymNameOrErr = SymI->getName();
  if (!SymNameOrErr)
    return SymNameOrErr.takeError();

  std::string SymName =
      Demangle ? demangle(*SymNameOrErr) : SymNameOrErr->str();
  if (SymbolDescription)
    SymName = getXCOFFSymbolDescription(createSymbolInfo(Obj, *SymI), SymName);

  Result.append(SymName.begin(), SymName.end());
  return Error::success();
}

std::optional<XCOFF::StorageMappingClass>
objdump::getXCOFFSymbolCsectSMC(const XCOFFObjectFile &Obj,
                                const SymbolRef &Sym) {
  const XCOFFSymbolRef SymRef = Obj.toSymbolRef(Sym.getRawDataRefImpl());

  if (!SymRef.isCsectSymbol())
    return std::nullopt;

  auto CsectAuxEntOrErr = SymRef.getXCOFFCsectAuxRef();
  if (!CsectAuxEntOrErr)
    return std::nullopt;

  return CsectAuxEntOrErr.get().getStorageMappingClass();
}

std::optional<object::SymbolRef>
objdump::getXCOFFSymbolContainingSymbolRef(const XCOFFObjectFile &Obj,
                                           const SymbolRef &Sym) {
  const XCOFFSymbolRef SymRef = Obj.toSymbolRef(Sym.getRawDataRefImpl());
  if (!SymRef.isCsectSymbol())
    return std::nullopt;

  Expected<XCOFFCsectAuxRef> CsectAuxEntOrErr = SymRef.getXCOFFCsectAuxRef();
  if (!CsectAuxEntOrErr || !CsectAuxEntOrErr.get().isLabel())
    return std::nullopt;
  uint32_t Idx =
      static_cast<uint32_t>(CsectAuxEntOrErr.get().getSectionOrLength());
  DataRefImpl DRI;
  DRI.p = Obj.getSymbolByIndex(Idx);
  return SymbolRef(DRI, &Obj);
}

bool objdump::isLabel(const XCOFFObjectFile &Obj, const SymbolRef &Sym) {
  const XCOFFSymbolRef SymRef = Obj.toSymbolRef(Sym.getRawDataRefImpl());
  if (!SymRef.isCsectSymbol())
    return false;

  auto CsectAuxEntOrErr = SymRef.getXCOFFCsectAuxRef();
  if (!CsectAuxEntOrErr)
    return false;

  return CsectAuxEntOrErr.get().isLabel();
}

std::string objdump::getXCOFFSymbolDescription(const SymbolInfoTy &SymbolInfo,
                                               StringRef SymbolName) {
  assert(SymbolInfo.isXCOFF() && "Must be a XCOFFSymInfo.");

  std::string Result;
  // Dummy symbols have no symbol index.
  if (SymbolInfo.XCOFFSymInfo.Index)
    Result =
        ("(idx: " + Twine(*SymbolInfo.XCOFFSymInfo.Index) + ") " + SymbolName)
            .str();
  else
    Result.append(SymbolName.begin(), SymbolName.end());

  if (SymbolInfo.XCOFFSymInfo.StorageMappingClass &&
      !SymbolInfo.XCOFFSymInfo.IsLabel) {
    const XCOFF::StorageMappingClass Smc =
        *SymbolInfo.XCOFFSymInfo.StorageMappingClass;
    Result.append(("[" + XCOFF::getMappingClassString(Smc) + "]").str());
  }

  return Result;
}

#define PRINTBOOL(Prefix, Obj, Field)                                          \
  OS << Prefix << " " << ((Obj.Field()) ? "+" : "-") << #Field

#define PRINTGET(Prefix, Obj, Field)                                           \
  OS << Prefix << " " << #Field << " = "                                       \
     << static_cast<unsigned>(Obj.get##Field())

#define PRINTOPTIONAL(Field)                                                   \
  if (TbTable.get##Field()) {                                                  \
    OS << '\n';                                                                \
    printRawData(Bytes.slice(Index, 4), Address + Index, OS, STI);             \
    Index += 4;                                                                \
    OS << "\t# " << #Field << " = " << *TbTable.get##Field();                  \
  }

void objdump::dumpTracebackTable(ArrayRef<uint8_t> Bytes, uint64_t Address,
                                 formatted_raw_ostream &OS, uint64_t End,
                                 const MCSubtargetInfo &STI,
                                 const XCOFFObjectFile *Obj) {
  uint64_t Index = 0;
  unsigned TabStop = getInstStartColumn(STI) - 1;
  // Print traceback table boundary.
  printRawData(Bytes.slice(Index, 4), Address, OS, STI);
  OS << "\t# Traceback table start\n";
  Index += 4;

  uint64_t Size = End - Address;
  bool Is64Bit = Obj->is64Bit();

  // XCOFFTracebackTable::create modifies the size parameter, so ensure Size
  // isn't changed.
  uint64_t SizeCopy = End - Address;
  Expected<XCOFFTracebackTable> TTOrErr =
      XCOFFTracebackTable::create(Bytes.data() + Index, SizeCopy, Is64Bit);

  if (!TTOrErr) {
    std::string WarningMsgStr;
    raw_string_ostream WarningStream(WarningMsgStr);
    WarningStream << "failure parsing traceback table with address: 0x"
                  << utohexstr(Address) + "\n>>> "
                  << toString(TTOrErr.takeError())
                  << "\n>>> Raw traceback table data is:\n";

    uint64_t LastNonZero = Index;
    for (uint64_t I = Index; I < Size; I += 4)
      if (support::endian::read32be(Bytes.slice(I, 4).data()) != 0)
        LastNonZero = I + 4 > Size ? Size : I + 4;

    if (Size - LastNonZero <= 4)
      LastNonZero = Size;

    formatted_raw_ostream FOS(WarningStream);
    while (Index < LastNonZero) {
      printRawData(Bytes.slice(Index, 4), Address + Index, FOS, STI);
      Index += 4;
      WarningStream << '\n';
    }

    // Print all remaining zeroes as ...
    if (Size - LastNonZero >= 8)
      WarningStream << "\t\t...\n";

    reportWarning(WarningMsgStr, Obj->getFileName());
    return;
  }

  auto PrintBytes = [&](uint64_t N) {
    printRawData(Bytes.slice(Index, N), Address + Index, OS, STI);
    Index += N;
  };

  XCOFFTracebackTable TbTable = *TTOrErr;
  // Print the first of the 8 bytes of mandatory fields.
  PrintBytes(1);
  OS << format("\t# Version = %i", TbTable.getVersion()) << '\n';

  // Print the second of the 8 bytes of mandatory fields.
  PrintBytes(1);
  TracebackTable::LanguageID LangId =
      static_cast<TracebackTable::LanguageID>(TbTable.getLanguageID());
  OS << "\t# Language = " << getNameForTracebackTableLanguageId(LangId) << '\n';

  auto Split = [&]() {
    OS << '\n';
    OS.indent(TabStop);
  };

  // Print the third of the 8 bytes of mandatory fields.
  PrintBytes(1);
  PRINTBOOL("\t#", TbTable, isGlobalLinkage);
  PRINTBOOL(",", TbTable, isOutOfLineEpilogOrPrologue);
  Split();
  PRINTBOOL("\t ", TbTable, hasTraceBackTableOffset);
  PRINTBOOL(",", TbTable, isInternalProcedure);
  Split();
  PRINTBOOL("\t ", TbTable, hasControlledStorage);
  PRINTBOOL(",", TbTable, isTOCless);
  Split();
  PRINTBOOL("\t ", TbTable, isFloatingPointPresent);
  Split();
  PRINTBOOL("\t ", TbTable, isFloatingPointOperationLogOrAbortEnabled);
  OS << '\n';

  // Print the 4th of the 8 bytes of mandatory fields.
  PrintBytes(1);
  PRINTBOOL("\t#", TbTable, isInterruptHandler);
  PRINTBOOL(",", TbTable, isFuncNamePresent);
  PRINTBOOL(",", TbTable, isAllocaUsed);
  Split();
  PRINTGET("\t ", TbTable, OnConditionDirective);
  PRINTBOOL(",", TbTable, isCRSaved);
  PRINTBOOL(",", TbTable, isLRSaved);
  OS << '\n';

  // Print the 5th of the 8 bytes of mandatory fields.
  PrintBytes(1);
  PRINTBOOL("\t#", TbTable, isBackChainStored);
  PRINTBOOL(",", TbTable, isFixup);
  PRINTGET(",", TbTable, NumOfFPRsSaved);
  OS << '\n';

  // Print the 6th of the 8 bytes of mandatory fields.
  PrintBytes(1);
  PRINTBOOL("\t#", TbTable, hasExtensionTable);
  PRINTBOOL(",", TbTable, hasVectorInfo);
  PRINTGET(",", TbTable, NumOfGPRsSaved);
  OS << '\n';

  // Print the 7th of the 8 bytes of mandatory fields.
  PrintBytes(1);
  PRINTGET("\t#", TbTable, NumberOfFixedParms);
  OS << '\n';

  // Print the 8th of the 8 bytes of mandatory fields.
  PrintBytes(1);
  PRINTGET("\t#", TbTable, NumberOfFPParms);
  PRINTBOOL(",", TbTable, hasParmsOnStack);

  PRINTOPTIONAL(ParmsType);
  PRINTOPTIONAL(TraceBackTableOffset);
  PRINTOPTIONAL(HandlerMask);
  PRINTOPTIONAL(NumOfCtlAnchors);

  if (TbTable.getControlledStorageInfoDisp()) {
    SmallVector<uint32_t, 8> Disp = *TbTable.getControlledStorageInfoDisp();
    for (unsigned I = 0; I < Disp.size(); ++I) {
      OS << '\n';
      PrintBytes(4);
      OS << "\t" << (I ? " " : "#") << " ControlledStorageInfoDisp[" << I
         << "] = " << Disp[I];
    }
  }

  // If there is a name, print the function name and function name length.
  if (TbTable.isFuncNamePresent()) {
    uint16_t FunctionNameLen = TbTable.getFunctionName()->size();
    if (FunctionNameLen == 0) {
      OS << '\n';
      reportWarning(
          "the length of the function name must be greater than zero if the "
          "isFuncNamePresent bit is set in the traceback table",
          Obj->getFileName());
      return;
    }

    OS << '\n';
    PrintBytes(2);
    OS << "\t# FunctionNameLen = " << FunctionNameLen;

    uint16_t RemainingBytes = FunctionNameLen;
    bool HasPrinted = false;
    while (RemainingBytes > 0) {
      OS << '\n';
      uint16_t PrintLen = RemainingBytes >= 4 ? 4 : RemainingBytes;
      printRawData(Bytes.slice(Index, PrintLen), Address + Index, OS, STI);
      Index += PrintLen;
      RemainingBytes -= PrintLen;

      if (!HasPrinted) {
        OS << "\t# FunctionName = " << *TbTable.getFunctionName();
        HasPrinted = true;
      }
    }
  }

  if (TbTable.isAllocaUsed()) {
    OS << '\n';
    PrintBytes(1);
    OS << format("\t# AllocaRegister = %u", *TbTable.getAllocaRegister());
  }

  if (TbTable.getVectorExt()) {
    OS << '\n';
    TBVectorExt VecExt = *TbTable.getVectorExt();
    // Print first byte of VectorExt.
    PrintBytes(1);
    PRINTGET("\t#", VecExt, NumberOfVRSaved);
    PRINTBOOL(",", VecExt, isVRSavedOnStack);
    PRINTBOOL(",", VecExt, hasVarArgs);
    OS << '\n';

    // Print the second byte of VectorExt.
    PrintBytes(1);
    PRINTGET("\t#", VecExt, NumberOfVectorParms);
    PRINTBOOL(",", VecExt, hasVMXInstruction);
    OS << '\n';

    PrintBytes(4);
    OS << "\t# VectorParmsInfoString = " << VecExt.getVectorParmsInfo();

    // There are two bytes of padding after vector info.
    OS << '\n';
    PrintBytes(2);
    OS << "\t# Padding";
  }

  if (TbTable.getExtensionTable()) {
    OS << '\n';
    PrintBytes(1);
    ExtendedTBTableFlag Flag =
        static_cast<ExtendedTBTableFlag>(*TbTable.getExtensionTable());
    OS << "\t# ExtensionTable = " << getExtendedTBTableFlagString(Flag);
  }

  if (TbTable.getEhInfoDisp()) {
    // There are 4 bytes alignment before eh info displacement.
    if (Index % 4) {
      OS << '\n';
      PrintBytes(4 - Index % 4);
      OS << "\t# Alignment padding for eh info displacement";
    }
    OS << '\n';
    // The size of the displacement (address) is 4 bytes in 32-bit object files,
    // and 8 bytes in 64-bit object files.
    PrintBytes(4);
    OS << "\t# EH info displacement";
    if (Is64Bit) {
      OS << '\n';
      PrintBytes(4);
    }
  }

  OS << '\n';
  if (End == Address + Index)
    return;

  Size = End - Address;

  const char *LineSuffix = "\t# Padding\n";
  auto IsWordZero = [&](uint64_t WordPos) {
    if (WordPos >= Size)
      return false;
    uint64_t LineLength = std::min(4 - WordPos % 4, Size - WordPos);
    return std::all_of(Bytes.begin() + WordPos,
                       Bytes.begin() + WordPos + LineLength,
                       [](uint8_t Byte) { return Byte == 0; });
  };

  bool AreWordsZero[] = {IsWordZero(Index), IsWordZero(alignTo(Index, 4) + 4),
                         IsWordZero(alignTo(Index, 4) + 8)};
  bool ShouldPrintLine = true;
  while (true) {
    // Determine the length of the line (4, except for the first line, which
    // will be just enough to align to the word boundary, and the last line,
    // which will be the remainder of the data).
    uint64_t LineLength = std::min(4 - Index % 4, Size - Index);
    if (ShouldPrintLine) {
      // Print the line.
      printRawData(Bytes.slice(Index, LineLength), Address + Index, OS, STI);
      OS << LineSuffix;
      LineSuffix = "\n";
    }

    Index += LineLength;
    if (Index == Size)
      return;

    // For 3 or more consecutive lines of zeros, skip all but the first one, and
    // replace them with "...".
    if (AreWordsZero[0] && AreWordsZero[1] && AreWordsZero[2]) {
      if (ShouldPrintLine)
        OS << std::string(8, ' ') << "...\n";
      ShouldPrintLine = false;
    } else if (!AreWordsZero[1]) {
      // We have reached the end of a skipped block of zeros.
      ShouldPrintLine = true;
    }
    AreWordsZero[0] = AreWordsZero[1];
    AreWordsZero[1] = AreWordsZero[2];
    AreWordsZero[2] = IsWordZero(Index + 8);
  }
}
#undef PRINTBOOL
#undef PRINTGET
#undef PRINTOPTIONAL
