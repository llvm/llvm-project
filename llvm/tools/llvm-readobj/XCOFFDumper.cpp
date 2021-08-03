//===-- XCOFFDumper.cpp - XCOFF dumping utility -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an XCOFF specific dumper for llvm-readobj.
//
//===----------------------------------------------------------------------===//

#include "ObjDumper.h"
#include "llvm-readobj.h"
#include "llvm/Object/XCOFFObjectFile.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace llvm;
using namespace object;

namespace {

class XCOFFDumper : public ObjDumper {

public:
  XCOFFDumper(const XCOFFObjectFile &Obj, ScopedPrinter &Writer)
      : ObjDumper(Writer, Obj.getFileName()), Obj(Obj) {}

  void printFileHeaders() override;
  void printSectionHeaders() override;
  void printRelocations() override;
  void printSymbols() override;
  void printDynamicSymbols() override;
  void printUnwindInfo() override;
  void printStackMap() const override;
  void printNeededLibraries() override;
  void printStringTable() override;

private:
  template <typename T> void printSectionHeaders(ArrayRef<T> Sections);
  template <typename T> void printGenericSectionHeader(T &Sec) const;
  template <typename T> void printOverflowSectionHeader(T &Sec) const;
  void printFileAuxEnt(const XCOFFFileAuxEnt *AuxEntPtr);
  void printCsectAuxEnt(XCOFFCsectAuxRef AuxEntRef);
  void printSectAuxEntForStat(const XCOFFSectAuxEntForStat *AuxEntPtr);
  void printSymbol(const SymbolRef &);
  void printRelocations(ArrayRef<XCOFFSectionHeader32> Sections);
  const XCOFFObjectFile &Obj;
};
} // anonymous namespace

void XCOFFDumper::printFileHeaders() {
  DictScope DS(W, "FileHeader");
  W.printHex("Magic", Obj.getMagic());
  W.printNumber("NumberOfSections", Obj.getNumberOfSections());

  // Negative timestamp values are reserved for future use.
  int32_t TimeStamp = Obj.getTimeStamp();
  if (TimeStamp > 0) {
    // This handling of the time stamp assumes that the host system's time_t is
    // compatible with AIX time_t. If a platform is not compatible, the lit
    // tests will let us know.
    time_t TimeDate = TimeStamp;

    char FormattedTime[21] = {};
    size_t BytesWritten =
        strftime(FormattedTime, 21, "%Y-%m-%dT%H:%M:%SZ", gmtime(&TimeDate));
    if (BytesWritten)
      W.printHex("TimeStamp", FormattedTime, TimeStamp);
    else
      W.printHex("Timestamp", TimeStamp);
  } else {
    W.printHex("TimeStamp", TimeStamp == 0 ? "None" : "Reserved Value",
               TimeStamp);
  }

  // The number of symbol table entries is an unsigned value in 64-bit objects
  // and a signed value (with negative values being 'reserved') in 32-bit
  // objects.
  if (Obj.is64Bit()) {
    W.printHex("SymbolTableOffset", Obj.getSymbolTableOffset64());
    W.printNumber("SymbolTableEntries", Obj.getNumberOfSymbolTableEntries64());
  } else {
    W.printHex("SymbolTableOffset", Obj.getSymbolTableOffset32());
    int32_t SymTabEntries = Obj.getRawNumberOfSymbolTableEntries32();
    if (SymTabEntries >= 0)
      W.printNumber("SymbolTableEntries", SymTabEntries);
    else
      W.printHex("SymbolTableEntries", "Reserved Value", SymTabEntries);
  }

  W.printHex("OptionalHeaderSize", Obj.getOptionalHeaderSize());
  W.printHex("Flags", Obj.getFlags());

  // TODO FIXME Add support for the auxiliary header (if any) once
  // XCOFFObjectFile has the necessary support.
}

void XCOFFDumper::printSectionHeaders() {
  if (Obj.is64Bit())
    printSectionHeaders(Obj.sections64());
  else
    printSectionHeaders(Obj.sections32());
}

void XCOFFDumper::printRelocations() {
  if (Obj.is64Bit())
    llvm_unreachable("64-bit relocation output not implemented!");
  else
    printRelocations(Obj.sections32());
}

static const EnumEntry<XCOFF::RelocationType> RelocationTypeNameclass[] = {
#define ECase(X)                                                               \
  { #X, XCOFF::X }
    ECase(R_POS),    ECase(R_RL),     ECase(R_RLA),    ECase(R_NEG),
    ECase(R_REL),    ECase(R_TOC),    ECase(R_TRL),    ECase(R_TRLA),
    ECase(R_GL),     ECase(R_TCL),    ECase(R_REF),    ECase(R_BA),
    ECase(R_BR),     ECase(R_RBA),    ECase(R_RBR),    ECase(R_TLS),
    ECase(R_TLS_IE), ECase(R_TLS_LD), ECase(R_TLS_LE), ECase(R_TLSM),
    ECase(R_TLSML),  ECase(R_TOCU),   ECase(R_TOCL)
#undef ECase
};

void XCOFFDumper::printRelocations(ArrayRef<XCOFFSectionHeader32> Sections) {
  if (!opts::ExpandRelocs)
    report_fatal_error("Unexpanded relocation output not implemented.");

  ListScope LS(W, "Relocations");
  uint16_t Index = 0;
  for (const auto &Sec : Sections) {
    ++Index;
    // Only the .text, .data, .tdata, and STYP_DWARF sections have relocation.
    if (Sec.Flags != XCOFF::STYP_TEXT && Sec.Flags != XCOFF::STYP_DATA &&
        Sec.Flags != XCOFF::STYP_TDATA && Sec.Flags != XCOFF::STYP_DWARF)
      continue;
    auto Relocations = unwrapOrError(Obj.getFileName(), Obj.relocations(Sec));
    if (Relocations.empty())
      continue;

    W.startLine() << "Section (index: " << Index << ") " << Sec.getName()
                  << " {\n";
    for (auto Reloc : Relocations) {
      StringRef SymbolName = unwrapOrError(
          Obj.getFileName(), Obj.getSymbolNameByIndex(Reloc.SymbolIndex));

      DictScope RelocScope(W, "Relocation");
      W.printHex("Virtual Address", Reloc.VirtualAddress);
      W.printNumber("Symbol", SymbolName, Reloc.SymbolIndex);
      W.printString("IsSigned", Reloc.isRelocationSigned() ? "Yes" : "No");
      W.printNumber("FixupBitValue", Reloc.isFixupIndicated() ? 1 : 0);
      W.printNumber("Length", Reloc.getRelocatedLength());
      W.printEnum("Type", (uint8_t)Reloc.Type,
                  makeArrayRef(RelocationTypeNameclass));
    }
    W.unindent();
    W.startLine() << "}\n";
  }
}

static const EnumEntry<XCOFF::CFileStringType> FileStringType[] = {
#define ECase(X)                                                               \
  { #X, XCOFF::X }
    ECase(XFT_FN), ECase(XFT_CT), ECase(XFT_CV), ECase(XFT_CD)
#undef ECase
};

static const EnumEntry<XCOFF::SymbolAuxType> SymAuxType[] = {
#define ECase(X)                                                               \
  { #X, XCOFF::X }
    ECase(AUX_EXCEPT), ECase(AUX_FCN), ECase(AUX_SYM), ECase(AUX_FILE),
    ECase(AUX_CSECT),  ECase(AUX_SECT)
#undef ECase
};

void XCOFFDumper::printFileAuxEnt(const XCOFFFileAuxEnt *AuxEntPtr) {
  assert((!Obj.is64Bit() || AuxEntPtr->AuxType == XCOFF::AUX_FILE) &&
         "Mismatched auxiliary type!");
  StringRef FileName =
      unwrapOrError(Obj.getFileName(), Obj.getCFileName(AuxEntPtr));
  DictScope SymDs(W, "File Auxiliary Entry");
  W.printNumber("Index",
                Obj.getSymbolIndex(reinterpret_cast<uintptr_t>(AuxEntPtr)));
  W.printString("Name", FileName);
  W.printEnum("Type", static_cast<uint8_t>(AuxEntPtr->Type),
              makeArrayRef(FileStringType));
  if (Obj.is64Bit()) {
    W.printEnum("Auxiliary Type", static_cast<uint8_t>(AuxEntPtr->AuxType),
                makeArrayRef(SymAuxType));
  }
}

static const EnumEntry<XCOFF::StorageMappingClass> CsectStorageMappingClass[] =
    {
#define ECase(X)                                                               \
  { #X, XCOFF::X }
        ECase(XMC_PR), ECase(XMC_RO), ECase(XMC_DB),   ECase(XMC_GL),
        ECase(XMC_XO), ECase(XMC_SV), ECase(XMC_SV64), ECase(XMC_SV3264),
        ECase(XMC_TI), ECase(XMC_TB), ECase(XMC_RW),   ECase(XMC_TC0),
        ECase(XMC_TC), ECase(XMC_TD), ECase(XMC_DS),   ECase(XMC_UA),
        ECase(XMC_BS), ECase(XMC_UC), ECase(XMC_TL),   ECase(XMC_UL),
        ECase(XMC_TE)
#undef ECase
};

static const EnumEntry<XCOFF::SymbolType> CsectSymbolTypeClass[] = {
#define ECase(X)                                                               \
  { #X, XCOFF::X }
    ECase(XTY_ER), ECase(XTY_SD), ECase(XTY_LD), ECase(XTY_CM)
#undef ECase
};

void XCOFFDumper::printCsectAuxEnt(XCOFFCsectAuxRef AuxEntRef) {
  assert((!Obj.is64Bit() || AuxEntRef.getAuxType64() == XCOFF::AUX_CSECT) &&
         "Mismatched auxiliary type!");

  DictScope SymDs(W, "CSECT Auxiliary Entry");
  W.printNumber("Index", Obj.getSymbolIndex(AuxEntRef.getEntryAddress()));
  W.printNumber(AuxEntRef.isLabel() ? "ContainingCsectSymbolIndex"
                                    : "SectionLen",
                AuxEntRef.getSectionOrLength());
  W.printHex("ParameterHashIndex", AuxEntRef.getParameterHashIndex());
  W.printHex("TypeChkSectNum", AuxEntRef.getTypeChkSectNum());
  // Print out symbol alignment and type.
  W.printNumber("SymbolAlignmentLog2", AuxEntRef.getAlignmentLog2());
  W.printEnum("SymbolType", AuxEntRef.getSymbolType(),
              makeArrayRef(CsectSymbolTypeClass));
  W.printEnum("StorageMappingClass",
              static_cast<uint8_t>(AuxEntRef.getStorageMappingClass()),
              makeArrayRef(CsectStorageMappingClass));

  if (Obj.is64Bit()) {
    W.printEnum("Auxiliary Type", static_cast<uint8_t>(XCOFF::AUX_CSECT),
                makeArrayRef(SymAuxType));
  } else {
    W.printHex("StabInfoIndex", AuxEntRef.getStabInfoIndex32());
    W.printHex("StabSectNum", AuxEntRef.getStabSectNum32());
  }
}

void XCOFFDumper::printSectAuxEntForStat(
    const XCOFFSectAuxEntForStat *AuxEntPtr) {
  assert(!Obj.is64Bit() && "32-bit interface called on 64-bit object file.");

  DictScope SymDs(W, "Sect Auxiliary Entry For Stat");
  W.printNumber("Index",
                Obj.getSymbolIndex(reinterpret_cast<uintptr_t>(AuxEntPtr)));
  W.printNumber("SectionLength", AuxEntPtr->SectionLength);

  // Unlike the corresponding fields in the section header, NumberOfRelocEnt
  // and NumberOfLineNum do not handle values greater than 65535.
  W.printNumber("NumberOfRelocEnt", AuxEntPtr->NumberOfRelocEnt);
  W.printNumber("NumberOfLineNum", AuxEntPtr->NumberOfLineNum);
}

static const EnumEntry<XCOFF::StorageClass> SymStorageClass[] = {
#define ECase(X)                                                               \
  { #X, XCOFF::X }
    ECase(C_NULL),  ECase(C_AUTO),    ECase(C_EXT),     ECase(C_STAT),
    ECase(C_REG),   ECase(C_EXTDEF),  ECase(C_LABEL),   ECase(C_ULABEL),
    ECase(C_MOS),   ECase(C_ARG),     ECase(C_STRTAG),  ECase(C_MOU),
    ECase(C_UNTAG), ECase(C_TPDEF),   ECase(C_USTATIC), ECase(C_ENTAG),
    ECase(C_MOE),   ECase(C_REGPARM), ECase(C_FIELD),   ECase(C_BLOCK),
    ECase(C_FCN),   ECase(C_EOS),     ECase(C_FILE),    ECase(C_LINE),
    ECase(C_ALIAS), ECase(C_HIDDEN),  ECase(C_HIDEXT),  ECase(C_BINCL),
    ECase(C_EINCL), ECase(C_INFO),    ECase(C_WEAKEXT), ECase(C_DWARF),
    ECase(C_GSYM),  ECase(C_LSYM),    ECase(C_PSYM),    ECase(C_RSYM),
    ECase(C_RPSYM), ECase(C_STSYM),   ECase(C_TCSYM),   ECase(C_BCOMM),
    ECase(C_ECOML), ECase(C_ECOMM),   ECase(C_DECL),    ECase(C_ENTRY),
    ECase(C_FUN),   ECase(C_BSTAT),   ECase(C_ESTAT),   ECase(C_GTLS),
    ECase(C_STTLS), ECase(C_EFCN)
#undef ECase
};

static StringRef GetSymbolValueName(XCOFF::StorageClass SC) {
  switch (SC) {
  case XCOFF::C_EXT:
  case XCOFF::C_WEAKEXT:
  case XCOFF::C_HIDEXT:
  case XCOFF::C_STAT:
    return "Value (RelocatableAddress)";
  case XCOFF::C_FILE:
    return "Value (SymbolTableIndex)";
  case XCOFF::C_FCN:
  case XCOFF::C_BLOCK:
  case XCOFF::C_FUN:
  case XCOFF::C_STSYM:
  case XCOFF::C_BINCL:
  case XCOFF::C_EINCL:
  case XCOFF::C_INFO:
  case XCOFF::C_BSTAT:
  case XCOFF::C_LSYM:
  case XCOFF::C_PSYM:
  case XCOFF::C_RPSYM:
  case XCOFF::C_RSYM:
  case XCOFF::C_ECOML:
  case XCOFF::C_DWARF:
    assert(false && "This StorageClass for the symbol is not yet implemented.");
    return "";
  default:
    return "Value";
  }
}

static const EnumEntry<XCOFF::CFileLangId> CFileLangIdClass[] = {
#define ECase(X)                                                               \
  { #X, XCOFF::X }
    ECase(TB_C), ECase(TB_CPLUSPLUS)
#undef ECase
};

static const EnumEntry<XCOFF::CFileCpuId> CFileCpuIdClass[] = {
#define ECase(X)                                                               \
  { #X, XCOFF::X }
    ECase(TCPU_PPC64), ECase(TCPU_COM), ECase(TCPU_970)
#undef ECase
};

void XCOFFDumper::printSymbol(const SymbolRef &S) {
  DataRefImpl SymbolDRI = S.getRawDataRefImpl();
  XCOFFSymbolRef SymbolEntRef = Obj.toSymbolRef(SymbolDRI);

  uint8_t NumberOfAuxEntries = SymbolEntRef.getNumberOfAuxEntries();

  DictScope SymDs(W, "Symbol");

  StringRef SymbolName =
      unwrapOrError(Obj.getFileName(), SymbolEntRef.getName());

  W.printNumber("Index", Obj.getSymbolIndex(SymbolEntRef.getEntryAddress()));
  W.printString("Name", SymbolName);
  W.printHex(GetSymbolValueName(SymbolEntRef.getStorageClass()),
             SymbolEntRef.getValue());

  StringRef SectionName =
      unwrapOrError(Obj.getFileName(), Obj.getSymbolSectionName(SymbolEntRef));

  W.printString("Section", SectionName);
  if (SymbolEntRef.getStorageClass() == XCOFF::C_FILE) {
    W.printEnum("Source Language ID", SymbolEntRef.getLanguageIdForCFile(),
                makeArrayRef(CFileLangIdClass));
    W.printEnum("CPU Version ID", SymbolEntRef.getCPUTypeIddForCFile(),
                makeArrayRef(CFileCpuIdClass));
  } else
    W.printHex("Type", SymbolEntRef.getSymbolType());

  W.printEnum("StorageClass",
              static_cast<uint8_t>(SymbolEntRef.getStorageClass()),
              makeArrayRef(SymStorageClass));
  W.printNumber("NumberOfAuxEntries", NumberOfAuxEntries);

  if (NumberOfAuxEntries == 0)
    return;

  switch (SymbolEntRef.getStorageClass()) {
  case XCOFF::C_FILE:
    // If the symbol is C_FILE and has auxiliary entries...
    for (int I = 1; I <= NumberOfAuxEntries; I++) {
      uintptr_t AuxAddress = XCOFFObjectFile::getAdvancedSymbolEntryAddress(
          SymbolEntRef.getEntryAddress(), I);

      if (Obj.is64Bit() &&
          *Obj.getSymbolAuxType(AuxAddress) != XCOFF::SymbolAuxType::AUX_FILE) {
        W.startLine() << "!Unexpected raw auxiliary entry data:\n";
        W.startLine() << format_bytes(
                             ArrayRef<uint8_t>(
                                 reinterpret_cast<const uint8_t *>(AuxAddress),
                                 XCOFF::SymbolTableEntrySize),
                             0, XCOFF::SymbolTableEntrySize)
                      << "\n";
        continue;
      }

      const XCOFFFileAuxEnt *FileAuxEntPtr =
          reinterpret_cast<const XCOFFFileAuxEnt *>(AuxAddress);
#ifndef NDEBUG
      Obj.checkSymbolEntryPointer(reinterpret_cast<uintptr_t>(FileAuxEntPtr));
#endif
      printFileAuxEnt(FileAuxEntPtr);
    }
    break;
  case XCOFF::C_EXT:
  case XCOFF::C_WEAKEXT:
  case XCOFF::C_HIDEXT: {
    // If the symbol is for a function, and it has more than 1 auxiliary entry,
    // then one of them must be function auxiliary entry which we do not
    // support yet.
    if (SymbolEntRef.isFunction() && NumberOfAuxEntries >= 2)
      report_fatal_error("Function auxiliary entry printing is unimplemented.");

    // If there is more than 1 auxiliary entry, instead of printing out
    // error information, print out the raw Auxiliary entry.
    // For 32-bit object, print from first to the last - 1. The last one must be
    // a CSECT Auxiliary Entry.
    // For 64-bit object, print from first to last and skips if SymbolAuxType is
    // AUX_CSECT.
    for (int I = 1; I <= NumberOfAuxEntries; I++) {
      if (I == NumberOfAuxEntries && !Obj.is64Bit())
        break;

      uintptr_t AuxAddress = XCOFFObjectFile::getAdvancedSymbolEntryAddress(
          SymbolEntRef.getEntryAddress(), I);
      if (Obj.is64Bit() &&
          *Obj.getSymbolAuxType(AuxAddress) == XCOFF::SymbolAuxType::AUX_CSECT)
        continue;

      W.startLine() << "!Unexpected raw auxiliary entry data:\n";
      W.startLine() << format_bytes(
          ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(AuxAddress),
                            XCOFF::SymbolTableEntrySize));
    }

    auto ErrOrCsectAuxRef = SymbolEntRef.getXCOFFCsectAuxRef();
    if (!ErrOrCsectAuxRef)
      reportUniqueWarning(ErrOrCsectAuxRef.takeError());
    else
      printCsectAuxEnt(*ErrOrCsectAuxRef);

    break;
  }
  case XCOFF::C_STAT:
    if (NumberOfAuxEntries > 1)
      report_fatal_error(
          "C_STAT symbol should not have more than 1 auxiliary entry.");

    const XCOFFSectAuxEntForStat *StatAuxEntPtr;
    StatAuxEntPtr = reinterpret_cast<const XCOFFSectAuxEntForStat *>(
        XCOFFObjectFile::getAdvancedSymbolEntryAddress(
            SymbolEntRef.getEntryAddress(), 1));
#ifndef NDEBUG
    Obj.checkSymbolEntryPointer(reinterpret_cast<uintptr_t>(StatAuxEntPtr));
#endif
    printSectAuxEntForStat(StatAuxEntPtr);
    break;
  case XCOFF::C_DWARF:
  case XCOFF::C_BLOCK:
  case XCOFF::C_FCN:
    report_fatal_error("Symbol table entry printing for this storage class "
                       "type is unimplemented.");
    break;
  default:
    for (int i = 1; i <= NumberOfAuxEntries; i++) {
      W.startLine() << "!Unexpected raw auxiliary entry data:\n";
      W.startLine() << format_bytes(
          ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(
                                XCOFFObjectFile::getAdvancedSymbolEntryAddress(
                                    SymbolEntRef.getEntryAddress(), i)),
                            XCOFF::SymbolTableEntrySize));
    }
    break;
  }
}

void XCOFFDumper::printSymbols() {
  ListScope Group(W, "Symbols");
  for (const SymbolRef &S : Obj.symbols())
    printSymbol(S);
}

void XCOFFDumper::printStringTable() {
  DictScope DS(W, "StringTable");
  StringRef StrTable = Obj.getStringTable();
  // Print strings from the fifth byte, since the first four bytes contain the
  // length (in bytes) of the string table (including the length field).
  printAsStringList(StrTable, 4);
}

void XCOFFDumper::printDynamicSymbols() {
  llvm_unreachable("Unimplemented functionality for XCOFFDumper");
}

void XCOFFDumper::printUnwindInfo() {
  llvm_unreachable("Unimplemented functionality for XCOFFDumper");
}

void XCOFFDumper::printStackMap() const {
  llvm_unreachable("Unimplemented functionality for XCOFFDumper");
}

void XCOFFDumper::printNeededLibraries() {
  llvm_unreachable("Unimplemented functionality for XCOFFDumper");
}

static const EnumEntry<XCOFF::SectionTypeFlags> SectionTypeFlagsNames[] = {
#define ECase(X)                                                               \
  { #X, XCOFF::X }
    ECase(STYP_PAD),    ECase(STYP_DWARF), ECase(STYP_TEXT),
    ECase(STYP_DATA),   ECase(STYP_BSS),   ECase(STYP_EXCEPT),
    ECase(STYP_INFO),   ECase(STYP_TDATA), ECase(STYP_TBSS),
    ECase(STYP_LOADER), ECase(STYP_DEBUG), ECase(STYP_TYPCHK),
    ECase(STYP_OVRFLO)
#undef ECase
};

template <typename T>
void XCOFFDumper::printOverflowSectionHeader(T &Sec) const {
  if (Obj.is64Bit()) {
    reportWarning(make_error<StringError>("An 64-bit XCOFF object file may not "
                                          "contain an overflow section header.",
                                          object_error::parse_failed),
                  Obj.getFileName());
  }

  W.printString("Name", Sec.getName());
  W.printNumber("NumberOfRelocations", Sec.PhysicalAddress);
  W.printNumber("NumberOfLineNumbers", Sec.VirtualAddress);
  W.printHex("Size", Sec.SectionSize);
  W.printHex("RawDataOffset", Sec.FileOffsetToRawData);
  W.printHex("RelocationPointer", Sec.FileOffsetToRelocationInfo);
  W.printHex("LineNumberPointer", Sec.FileOffsetToLineNumberInfo);
  W.printNumber("IndexOfSectionOverflowed", Sec.NumberOfRelocations);
  W.printNumber("IndexOfSectionOverflowed", Sec.NumberOfLineNumbers);
}

template <typename T>
void XCOFFDumper::printGenericSectionHeader(T &Sec) const {
  W.printString("Name", Sec.getName());
  W.printHex("PhysicalAddress", Sec.PhysicalAddress);
  W.printHex("VirtualAddress", Sec.VirtualAddress);
  W.printHex("Size", Sec.SectionSize);
  W.printHex("RawDataOffset", Sec.FileOffsetToRawData);
  W.printHex("RelocationPointer", Sec.FileOffsetToRelocationInfo);
  W.printHex("LineNumberPointer", Sec.FileOffsetToLineNumberInfo);
  W.printNumber("NumberOfRelocations", Sec.NumberOfRelocations);
  W.printNumber("NumberOfLineNumbers", Sec.NumberOfLineNumbers);
}

template <typename T>
void XCOFFDumper::printSectionHeaders(ArrayRef<T> Sections) {
  ListScope Group(W, "Sections");

  uint16_t Index = 1;
  for (const T &Sec : Sections) {
    DictScope SecDS(W, "Section");

    W.printNumber("Index", Index++);
    uint16_t SectionType = Sec.getSectionType();
    switch (SectionType) {
    case XCOFF::STYP_OVRFLO:
      printOverflowSectionHeader(Sec);
      break;
    case XCOFF::STYP_LOADER:
    case XCOFF::STYP_EXCEPT:
    case XCOFF::STYP_TYPCHK:
      // TODO The interpretation of loader, exception and type check section
      // headers are different from that of generic section headers. We will
      // implement them later. We interpret them as generic section headers for
      // now.
    default:
      printGenericSectionHeader(Sec);
      break;
    }
    if (Sec.isReservedSectionType())
      W.printHex("Flags", "Reserved", SectionType);
    else
      W.printEnum("Type", SectionType, makeArrayRef(SectionTypeFlagsNames));
  }

  if (opts::SectionRelocations)
    report_fatal_error("Dumping section relocations is unimplemented");

  if (opts::SectionSymbols)
    report_fatal_error("Dumping symbols is unimplemented");

  if (opts::SectionData)
    report_fatal_error("Dumping section data is unimplemented");
}

namespace llvm {
std::unique_ptr<ObjDumper>
createXCOFFDumper(const object::XCOFFObjectFile &XObj, ScopedPrinter &Writer) {
  return std::make_unique<XCOFFDumper>(XObj, Writer);
}
} // namespace llvm
