//===------ xcoff2yaml.cpp - XCOFF YAMLIO implementation --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "obj2yaml.h"
#include "llvm/Object/XCOFFObjectFile.h"
#include "llvm/ObjectYAML/XCOFFYAML.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/YAMLTraits.h"

using namespace llvm;
using namespace llvm::object;
namespace {

class XCOFFDumper {
  const object::XCOFFObjectFile &Obj;
  XCOFFYAML::Object YAMLObj;
  void dumpHeader();
  Error dumpSections();
  Error dumpSymbols();
  template <typename Shdr, typename Reloc>
  Error dumpSections(ArrayRef<Shdr> Sections);

public:
  XCOFFDumper(const object::XCOFFObjectFile &obj) : Obj(obj) {}
  Error dump();
  XCOFFYAML::Object &getYAMLObj() { return YAMLObj; }

  template <typename T> const T *getAuxEntPtr(uintptr_t AuxAddress) {
    Obj.checkSymbolEntryPointer(AuxAddress);
    return reinterpret_cast<const T *>(AuxAddress);
  }
};
} // namespace

Error XCOFFDumper::dump() {
  dumpHeader();
  if (Error E = dumpSections())
    return E;
  return dumpSymbols();
}

void XCOFFDumper::dumpHeader() {
  YAMLObj.Header.Magic = Obj.getMagic();
  YAMLObj.Header.NumberOfSections = Obj.getNumberOfSections();
  YAMLObj.Header.TimeStamp = Obj.getTimeStamp();
  YAMLObj.Header.SymbolTableOffset = Obj.is64Bit()
                                         ? Obj.getSymbolTableOffset64()
                                         : Obj.getSymbolTableOffset32();
  YAMLObj.Header.NumberOfSymTableEntries =
      Obj.is64Bit() ? Obj.getNumberOfSymbolTableEntries64()
                    : Obj.getRawNumberOfSymbolTableEntries32();
  YAMLObj.Header.AuxHeaderSize = Obj.getOptionalHeaderSize();
  YAMLObj.Header.Flags = Obj.getFlags();
}

Error XCOFFDumper::dumpSections() {
  if (Obj.is64Bit())
    return dumpSections<XCOFFSectionHeader64, XCOFFRelocation64>(
        Obj.sections64());
  return dumpSections<XCOFFSectionHeader32, XCOFFRelocation32>(
      Obj.sections32());
}

template <typename Shdr, typename Reloc>
Error XCOFFDumper::dumpSections(ArrayRef<Shdr> Sections) {
  std::vector<XCOFFYAML::Section> &YamlSections = YAMLObj.Sections;
  for (const Shdr &S : Sections) {
    XCOFFYAML::Section YamlSec;
    YamlSec.SectionName = S.getName();
    YamlSec.Address = S.PhysicalAddress;
    YamlSec.Size = S.SectionSize;
    YamlSec.NumberOfRelocations = S.NumberOfRelocations;
    YamlSec.NumberOfLineNumbers = S.NumberOfLineNumbers;
    YamlSec.FileOffsetToData = S.FileOffsetToRawData;
    YamlSec.FileOffsetToRelocations = S.FileOffsetToRelocationInfo;
    YamlSec.FileOffsetToLineNumbers = S.FileOffsetToLineNumberInfo;
    YamlSec.Flags = S.Flags;

    // Dump section data.
    if (S.FileOffsetToRawData) {
      DataRefImpl SectionDRI;
      SectionDRI.p = reinterpret_cast<uintptr_t>(&S);
      Expected<ArrayRef<uint8_t>> SecDataRefOrErr =
          Obj.getSectionContents(SectionDRI);
      if (!SecDataRefOrErr)
        return SecDataRefOrErr.takeError();
      YamlSec.SectionData = SecDataRefOrErr.get();
    }

    // Dump relocations.
    if (S.NumberOfRelocations) {
      auto RelRefOrErr = Obj.relocations<Shdr, Reloc>(S);
      if (!RelRefOrErr)
        return RelRefOrErr.takeError();
      for (const Reloc &R : RelRefOrErr.get()) {
        XCOFFYAML::Relocation YamlRel;
        YamlRel.Type = R.Type;
        YamlRel.Info = R.Info;
        YamlRel.SymbolIndex = R.SymbolIndex;
        YamlRel.VirtualAddress = R.VirtualAddress;
        YamlSec.Relocations.push_back(YamlRel);
      }
    }
    YamlSections.push_back(YamlSec);
  }
  return Error::success();
}

static void
dumpFileAuxSym(std::vector<std::unique_ptr<XCOFFYAML::AuxSymbolEnt>> &AuxEntTbl,
               XCOFF::CFileStringType Type, StringRef FileStr) {
  XCOFFYAML::FileAuxEnt FileAuxSym;
  FileAuxSym.FileNameOrString = FileStr;
  FileAuxSym.FileStringType = Type;
  AuxEntTbl.push_back(std::make_unique<XCOFFYAML::FileAuxEnt>(FileAuxSym));
}

static void
dumpStatAuxSym(std::vector<std::unique_ptr<XCOFFYAML::AuxSymbolEnt>> &AuxEntTbl,
               const object::XCOFFSectAuxEntForStat &AuxEntPtr) {
  XCOFFYAML::SectAuxEntForStat StatAuxSym;
  StatAuxSym.SectionLength = AuxEntPtr.SectionLength;
  StatAuxSym.NumberOfLineNum = AuxEntPtr.NumberOfLineNum;
  StatAuxSym.NumberOfRelocEnt = AuxEntPtr.NumberOfRelocEnt;
  AuxEntTbl.push_back(
      std::make_unique<XCOFFYAML::SectAuxEntForStat>(StatAuxSym));
}

static void
dumpFunAuxSym(std::vector<std::unique_ptr<XCOFFYAML::AuxSymbolEnt>> &AuxEntTbl,
              const object::XCOFFFunctionAuxEnt32 &AuxEntPtr) {
  XCOFFYAML::FunctionAuxEnt FunAuxSym;
  FunAuxSym.OffsetToExceptionTbl = AuxEntPtr.OffsetToExceptionTbl;
  FunAuxSym.PtrToLineNum = AuxEntPtr.PtrToLineNum;
  FunAuxSym.SizeOfFunction = AuxEntPtr.SizeOfFunction;
  FunAuxSym.SymIdxOfNextBeyond = AuxEntPtr.SymIdxOfNextBeyond;
  AuxEntTbl.push_back(std::make_unique<XCOFFYAML::FunctionAuxEnt>(FunAuxSym));
}

static void
dumpFunAuxSym(std::vector<std::unique_ptr<XCOFFYAML::AuxSymbolEnt>> &AuxEntTbl,
              const object::XCOFFFunctionAuxEnt64 &AuxEntPtr) {
  XCOFFYAML::FunctionAuxEnt FunAuxSym;
  FunAuxSym.PtrToLineNum = AuxEntPtr.PtrToLineNum;
  FunAuxSym.SizeOfFunction = AuxEntPtr.SizeOfFunction;
  FunAuxSym.SymIdxOfNextBeyond = AuxEntPtr.SymIdxOfNextBeyond;
  AuxEntTbl.push_back(std::make_unique<XCOFFYAML::FunctionAuxEnt>(FunAuxSym));
}

static void
dumpExpAuxSym(std::vector<std::unique_ptr<XCOFFYAML::AuxSymbolEnt>> &AuxEntTbl,
              const object::XCOFFExceptionAuxEnt &AuxEntPtr) {
  XCOFFYAML::ExcpetionAuxEnt ExceptAuxSym;
  ExceptAuxSym.OffsetToExceptionTbl = AuxEntPtr.OffsetToExceptionTbl;
  ExceptAuxSym.SizeOfFunction = AuxEntPtr.SizeOfFunction;
  ExceptAuxSym.SymIdxOfNextBeyond = AuxEntPtr.SymIdxOfNextBeyond;
  AuxEntTbl.push_back(
      std::make_unique<XCOFFYAML::ExcpetionAuxEnt>(ExceptAuxSym));
}

static void dumpCscetAuxSym(
    std::vector<std::unique_ptr<XCOFFYAML::AuxSymbolEnt>> &AuxEntTbl,
    const object::XCOFFCsectAuxRef &AuxEntPtr, bool Is64bit) {
  XCOFFYAML::CsectAuxEnt CsectAuxSym;
  CsectAuxSym.ParameterHashIndex = AuxEntPtr.getParameterHashIndex();
  CsectAuxSym.TypeChkSectNum = AuxEntPtr.getTypeChkSectNum();
  CsectAuxSym.SymbolAlignmentAndType = AuxEntPtr.getSymbolAlignmentAndType();
  CsectAuxSym.StorageMappingClass = AuxEntPtr.getStorageMappingClass();
  if (Is64bit) {
    CsectAuxSym.SectionOrLengthLo =
        static_cast<uint32_t>(AuxEntPtr.getSectionOrLength64());
    CsectAuxSym.SectionOrLengthHi =
        static_cast<uint32_t>(AuxEntPtr.getSectionOrLength64() >> 32);
  } else {
    CsectAuxSym.SectionOrLength = AuxEntPtr.getSectionOrLength32();
    CsectAuxSym.StabInfoIndex = AuxEntPtr.getStabInfoIndex32();
    CsectAuxSym.StabSectNum = AuxEntPtr.getStabSectNum32();
  }
  AuxEntTbl.push_back(std::make_unique<XCOFFYAML::CsectAuxEnt>(CsectAuxSym));
}

Error XCOFFDumper::dumpSymbols() {
  std::vector<XCOFFYAML::Symbol> &Symbols = YAMLObj.Symbols;

  for (const SymbolRef &S : Obj.symbols()) {
    DataRefImpl SymbolDRI = S.getRawDataRefImpl();
    const XCOFFSymbolRef SymbolEntRef = Obj.toSymbolRef(SymbolDRI);
    XCOFFYAML::Symbol Sym;

    Expected<StringRef> SymNameRefOrErr = Obj.getSymbolName(SymbolDRI);
    if (!SymNameRefOrErr) {
      return SymNameRefOrErr.takeError();
    }
    Sym.SymbolName = SymNameRefOrErr.get();

    Sym.Value = SymbolEntRef.getValue();

    Expected<StringRef> SectionNameRefOrErr =
        Obj.getSymbolSectionName(SymbolEntRef);
    if (!SectionNameRefOrErr)
      return SectionNameRefOrErr.takeError();

    Sym.SectionName = SectionNameRefOrErr.get();

    Sym.Type = SymbolEntRef.getSymbolType();
    Sym.StorageClass = SymbolEntRef.getStorageClass();
    uint8_t NumOfAuxSym = SymbolEntRef.getNumberOfAuxEntries();
    Sym.NumberOfAuxEntries = NumOfAuxSym;

    if (NumOfAuxSym) {
      std::vector<std::unique_ptr<XCOFFYAML::AuxSymbolEnt>> AuxEntTbl;
      switch (Sym.StorageClass) {
      case XCOFF::C_FILE: {
        for (uint8_t I = 1; I <= NumOfAuxSym; ++I) {
          uintptr_t AuxAddress = XCOFFObjectFile::getAdvancedSymbolEntryAddress(
              SymbolEntRef.getEntryAddress(), I);
          const XCOFFFileAuxEnt *FileAuxEntPtr =
              getAuxEntPtr<XCOFFFileAuxEnt>(AuxAddress);
          auto FileNameOrError = Obj.getCFileName(FileAuxEntPtr);
          if (!FileNameOrError)
            return FileNameOrError.takeError();

          dumpFileAuxSym(AuxEntTbl, FileAuxEntPtr->Type, FileNameOrError.get());
        }
        break;
      }
      case XCOFF::C_STAT: {
        assert(NumOfAuxSym == 1 && "expected a single aux symbol for C_STAT!");
        const XCOFFSectAuxEntForStat *AuxEntPtr =
            getAuxEntPtr<XCOFFSectAuxEntForStat>(
                XCOFFObjectFile::getAdvancedSymbolEntryAddress(
                    SymbolEntRef.getEntryAddress(), 1));
        dumpStatAuxSym(AuxEntTbl, *AuxEntPtr);
        break;
      }
      case XCOFF::C_EXT:
      case XCOFF::C_WEAKEXT:
      case XCOFF::C_HIDEXT: {
        for (uint8_t I = 1; I <= NumOfAuxSym; ++I) {
          if (I == NumOfAuxSym && !Obj.is64Bit())
            break;

          uintptr_t AuxAddress = XCOFFObjectFile::getAdvancedSymbolEntryAddress(
              SymbolEntRef.getEntryAddress(), I);
          if (!Obj.is64Bit()) {
            const XCOFFFunctionAuxEnt32 *AuxEntPtr =
                getAuxEntPtr<XCOFFFunctionAuxEnt32>(AuxAddress);
            dumpFunAuxSym(AuxEntTbl, *AuxEntPtr);
          } else {
            XCOFF::SymbolAuxType Type = *Obj.getSymbolAuxType(AuxAddress);
            if (Type == XCOFF::SymbolAuxType::AUX_CSECT)
              continue;
            if (Type == XCOFF::SymbolAuxType::AUX_FCN) {
              const XCOFFFunctionAuxEnt64 *AuxEntPtr =
                  getAuxEntPtr<XCOFFFunctionAuxEnt64>(AuxAddress);
              dumpFunAuxSym(AuxEntTbl, *AuxEntPtr);
            } else if (Type == XCOFF::SymbolAuxType::AUX_EXCEPT) {
              const XCOFFExceptionAuxEnt *AuxEntPtr =
                  getAuxEntPtr<XCOFFExceptionAuxEnt>(AuxAddress);
              dumpExpAuxSym(AuxEntTbl, *AuxEntPtr);
            } else
              llvm_unreachable("invalid aux symbol entry");
          }
        }
        auto ErrOrCsectAuxRef = SymbolEntRef.getXCOFFCsectAuxRef();
        if (!ErrOrCsectAuxRef)
          return ErrOrCsectAuxRef.takeError();
        dumpCscetAuxSym(AuxEntTbl, ErrOrCsectAuxRef.get(), Obj.is64Bit());
        break;
      }
      case XCOFF::C_BLOCK:
      case XCOFF::C_FCN: {
        assert(NumOfAuxSym == 1 &&
               "expected a single aux symbol for C_BLOCK or C_FCN!");

        uintptr_t AuxAddress = XCOFFObjectFile::getAdvancedSymbolEntryAddress(
            SymbolEntRef.getEntryAddress(), 1);
        XCOFFYAML::BlockAuxEnt BlockAuxSym;
        if (Obj.is64Bit()) {
          const XCOFFBlockAuxEnt64 *AuxEntPtr =
              getAuxEntPtr<XCOFFBlockAuxEnt64>(AuxAddress);
          BlockAuxSym.LineNum = AuxEntPtr->LineNum;
        } else {
          const XCOFFBlockAuxEnt32 *AuxEntPtr =
              getAuxEntPtr<XCOFFBlockAuxEnt32>(AuxAddress);
          BlockAuxSym.LineNumLo = AuxEntPtr->LineNumLo;
          BlockAuxSym.LineNumHi = AuxEntPtr->LineNumHi;
        }
        AuxEntTbl.push_back(
            std::make_unique<XCOFFYAML::BlockAuxEnt>(BlockAuxSym));
        break;
      }
      case XCOFF::C_DWARF: {
        assert(NumOfAuxSym == 1 && "expected a single aux symbol for C_DWARF!");
        uintptr_t AuxAddress = XCOFFObjectFile::getAdvancedSymbolEntryAddress(
            SymbolEntRef.getEntryAddress(), 1);
        XCOFFYAML::SectAuxEntForDWARF DwarfAuxSym;
        if (Obj.is64Bit()) {
          const XCOFFSectAuxEntForDWARF64 *AuxEntPtr =
              getAuxEntPtr<XCOFFSectAuxEntForDWARF64>(AuxAddress);
          DwarfAuxSym.LengthOfSectionPortion =
              AuxEntPtr->LengthOfSectionPortion;
          DwarfAuxSym.NumberOfRelocEnt = AuxEntPtr->NumberOfRelocEnt;
        } else {
          const XCOFFSectAuxEntForDWARF32 *AuxEntPtr =
              getAuxEntPtr<XCOFFSectAuxEntForDWARF32>(AuxAddress);
          DwarfAuxSym.LengthOfSectionPortion =
              AuxEntPtr->LengthOfSectionPortion;
          DwarfAuxSym.NumberOfRelocEnt = AuxEntPtr->NumberOfRelocEnt;
        }
        AuxEntTbl.push_back(
            std::make_unique<XCOFFYAML::SectAuxEntForDWARF>(DwarfAuxSym));
        break;
      }
      default:
        break;
      }
      Sym.AuxEntries = std::move(AuxEntTbl);
    }

    Symbols.push_back(std::move(Sym));
  }

  return Error::success();
}

Error xcoff2yaml(raw_ostream &Out, const object::XCOFFObjectFile &Obj) {
  XCOFFDumper Dumper(Obj);

  if (Error E = Dumper.dump())
    return E;

  yaml::Output Yout(Out);
  Yout << Dumper.getYAMLObj();

  return Error::success();
}
