//===------ utils/obj2yaml.cpp - obj2yaml conversion tool -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "obj2yaml.h"

#include "llvm/Object/COFF.h"


template <typename One, typename Two>
struct pod_pair { // I'd much rather use std::pair, but it's not a POD
  One first;
  Two second;
};

#define STRING_PAIR(x)  {llvm::COFF::x, #x}
static const pod_pair<llvm::COFF::MachineTypes, const char *> 
MachineTypePairs [] = {
  STRING_PAIR(IMAGE_FILE_MACHINE_UNKNOWN),
  STRING_PAIR(IMAGE_FILE_MACHINE_AM33),
  STRING_PAIR(IMAGE_FILE_MACHINE_AMD64),
  STRING_PAIR(IMAGE_FILE_MACHINE_ARM),
  STRING_PAIR(IMAGE_FILE_MACHINE_ARMV7),
  STRING_PAIR(IMAGE_FILE_MACHINE_EBC),
  STRING_PAIR(IMAGE_FILE_MACHINE_I386),
  STRING_PAIR(IMAGE_FILE_MACHINE_IA64),
  STRING_PAIR(IMAGE_FILE_MACHINE_M32R),
  STRING_PAIR(IMAGE_FILE_MACHINE_MIPS16),
  STRING_PAIR(IMAGE_FILE_MACHINE_MIPSFPU),
  STRING_PAIR(IMAGE_FILE_MACHINE_MIPSFPU16),
  STRING_PAIR(IMAGE_FILE_MACHINE_POWERPC),
  STRING_PAIR(IMAGE_FILE_MACHINE_POWERPCFP),
  STRING_PAIR(IMAGE_FILE_MACHINE_R4000),
  STRING_PAIR(IMAGE_FILE_MACHINE_SH3),
  STRING_PAIR(IMAGE_FILE_MACHINE_SH3DSP),
  STRING_PAIR(IMAGE_FILE_MACHINE_SH4),
  STRING_PAIR(IMAGE_FILE_MACHINE_SH5),
  STRING_PAIR(IMAGE_FILE_MACHINE_THUMB),
  STRING_PAIR(IMAGE_FILE_MACHINE_WCEMIPSV2)
};

static const pod_pair<llvm::COFF::SectionCharacteristics, const char *> 
SectionCharacteristicsPairs1 [] = {
  STRING_PAIR(IMAGE_SCN_TYPE_NO_PAD),
  STRING_PAIR(IMAGE_SCN_CNT_CODE),
  STRING_PAIR(IMAGE_SCN_CNT_INITIALIZED_DATA),
  STRING_PAIR(IMAGE_SCN_CNT_UNINITIALIZED_DATA),
  STRING_PAIR(IMAGE_SCN_LNK_OTHER),
  STRING_PAIR(IMAGE_SCN_LNK_INFO),
  STRING_PAIR(IMAGE_SCN_LNK_REMOVE),
  STRING_PAIR(IMAGE_SCN_LNK_COMDAT),
  STRING_PAIR(IMAGE_SCN_GPREL),
  STRING_PAIR(IMAGE_SCN_MEM_PURGEABLE),
  STRING_PAIR(IMAGE_SCN_MEM_16BIT),
  STRING_PAIR(IMAGE_SCN_MEM_LOCKED),
  STRING_PAIR(IMAGE_SCN_MEM_PRELOAD)
};

static const pod_pair<llvm::COFF::SectionCharacteristics, const char *> 
SectionCharacteristicsPairsAlignment [] = {
  STRING_PAIR(IMAGE_SCN_ALIGN_1BYTES),
  STRING_PAIR(IMAGE_SCN_ALIGN_2BYTES),
  STRING_PAIR(IMAGE_SCN_ALIGN_4BYTES),
  STRING_PAIR(IMAGE_SCN_ALIGN_8BYTES),
  STRING_PAIR(IMAGE_SCN_ALIGN_16BYTES),
  STRING_PAIR(IMAGE_SCN_ALIGN_32BYTES),
  STRING_PAIR(IMAGE_SCN_ALIGN_64BYTES),
  STRING_PAIR(IMAGE_SCN_ALIGN_128BYTES),
  STRING_PAIR(IMAGE_SCN_ALIGN_256BYTES),
  STRING_PAIR(IMAGE_SCN_ALIGN_512BYTES),
  STRING_PAIR(IMAGE_SCN_ALIGN_1024BYTES),
  STRING_PAIR(IMAGE_SCN_ALIGN_2048BYTES),
  STRING_PAIR(IMAGE_SCN_ALIGN_4096BYTES),
  STRING_PAIR(IMAGE_SCN_ALIGN_8192BYTES)
};

static const pod_pair<llvm::COFF::SectionCharacteristics, const char *> 
SectionCharacteristicsPairs2 [] = {
  STRING_PAIR(IMAGE_SCN_LNK_NRELOC_OVFL),
  STRING_PAIR(IMAGE_SCN_MEM_DISCARDABLE),
  STRING_PAIR(IMAGE_SCN_MEM_NOT_CACHED),
  STRING_PAIR(IMAGE_SCN_MEM_NOT_PAGED),
  STRING_PAIR(IMAGE_SCN_MEM_SHARED),
  STRING_PAIR(IMAGE_SCN_MEM_EXECUTE),
  STRING_PAIR(IMAGE_SCN_MEM_READ),
  STRING_PAIR(IMAGE_SCN_MEM_WRITE)
};
  
static const pod_pair<llvm::COFF::SymbolBaseType, const char *> 
SymbolBaseTypePairs [] = {
  STRING_PAIR(IMAGE_SYM_TYPE_NULL),
  STRING_PAIR(IMAGE_SYM_TYPE_VOID),
  STRING_PAIR(IMAGE_SYM_TYPE_CHAR),
  STRING_PAIR(IMAGE_SYM_TYPE_SHORT),
  STRING_PAIR(IMAGE_SYM_TYPE_INT),
  STRING_PAIR(IMAGE_SYM_TYPE_LONG),
  STRING_PAIR(IMAGE_SYM_TYPE_FLOAT),
  STRING_PAIR(IMAGE_SYM_TYPE_DOUBLE),
  STRING_PAIR(IMAGE_SYM_TYPE_STRUCT),
  STRING_PAIR(IMAGE_SYM_TYPE_UNION),
  STRING_PAIR(IMAGE_SYM_TYPE_ENUM),
  STRING_PAIR(IMAGE_SYM_TYPE_MOE),
  STRING_PAIR(IMAGE_SYM_TYPE_BYTE),
  STRING_PAIR(IMAGE_SYM_TYPE_WORD),
  STRING_PAIR(IMAGE_SYM_TYPE_UINT),
  STRING_PAIR(IMAGE_SYM_TYPE_DWORD)
};

static const pod_pair<llvm::COFF::SymbolComplexType, const char *> 
SymbolComplexTypePairs [] = {
  STRING_PAIR(IMAGE_SYM_DTYPE_NULL),
  STRING_PAIR(IMAGE_SYM_DTYPE_POINTER),
  STRING_PAIR(IMAGE_SYM_DTYPE_FUNCTION),
  STRING_PAIR(IMAGE_SYM_DTYPE_ARRAY),
};
  
static const pod_pair<llvm::COFF::SymbolStorageClass, const char *> 
SymbolStorageClassPairs [] = {
  STRING_PAIR(IMAGE_SYM_CLASS_END_OF_FUNCTION),
  STRING_PAIR(IMAGE_SYM_CLASS_NULL),
  STRING_PAIR(IMAGE_SYM_CLASS_AUTOMATIC),
  STRING_PAIR(IMAGE_SYM_CLASS_EXTERNAL),
  STRING_PAIR(IMAGE_SYM_CLASS_STATIC),
  STRING_PAIR(IMAGE_SYM_CLASS_REGISTER),
  STRING_PAIR(IMAGE_SYM_CLASS_EXTERNAL_DEF),
  STRING_PAIR(IMAGE_SYM_CLASS_LABEL),
  STRING_PAIR(IMAGE_SYM_CLASS_UNDEFINED_LABEL),
  STRING_PAIR(IMAGE_SYM_CLASS_MEMBER_OF_STRUCT),
  STRING_PAIR(IMAGE_SYM_CLASS_ARGUMENT),
  STRING_PAIR(IMAGE_SYM_CLASS_STRUCT_TAG),
  STRING_PAIR(IMAGE_SYM_CLASS_MEMBER_OF_UNION),
  STRING_PAIR(IMAGE_SYM_CLASS_UNION_TAG),
  STRING_PAIR(IMAGE_SYM_CLASS_TYPE_DEFINITION),
  STRING_PAIR(IMAGE_SYM_CLASS_UNDEFINED_STATIC),
  STRING_PAIR(IMAGE_SYM_CLASS_ENUM_TAG),
  STRING_PAIR(IMAGE_SYM_CLASS_MEMBER_OF_ENUM),
  STRING_PAIR(IMAGE_SYM_CLASS_REGISTER_PARAM),
  STRING_PAIR(IMAGE_SYM_CLASS_BIT_FIELD),
  STRING_PAIR(IMAGE_SYM_CLASS_BLOCK),
  STRING_PAIR(IMAGE_SYM_CLASS_FUNCTION),
  STRING_PAIR(IMAGE_SYM_CLASS_END_OF_STRUCT),
  STRING_PAIR(IMAGE_SYM_CLASS_FILE),
  STRING_PAIR(IMAGE_SYM_CLASS_SECTION),
  STRING_PAIR(IMAGE_SYM_CLASS_WEAK_EXTERNAL),
  STRING_PAIR(IMAGE_SYM_CLASS_CLR_TOKEN),
};

static const pod_pair<llvm::COFF::RelocationTypeX86, const char *> 
RelocationTypeX86Pairs [] = {
  STRING_PAIR(IMAGE_REL_I386_ABSOLUTE),
  STRING_PAIR(IMAGE_REL_I386_DIR16),
  STRING_PAIR(IMAGE_REL_I386_REL16),
  STRING_PAIR(IMAGE_REL_I386_DIR32),
  STRING_PAIR(IMAGE_REL_I386_DIR32NB),
  STRING_PAIR(IMAGE_REL_I386_SEG12),
  STRING_PAIR(IMAGE_REL_I386_SECTION),
  STRING_PAIR(IMAGE_REL_I386_SECREL),
  STRING_PAIR(IMAGE_REL_I386_TOKEN),
  STRING_PAIR(IMAGE_REL_I386_SECREL7),
  STRING_PAIR(IMAGE_REL_I386_REL32),
  STRING_PAIR(IMAGE_REL_AMD64_ABSOLUTE),
  STRING_PAIR(IMAGE_REL_AMD64_ADDR64),
  STRING_PAIR(IMAGE_REL_AMD64_ADDR32),
  STRING_PAIR(IMAGE_REL_AMD64_ADDR32NB),
  STRING_PAIR(IMAGE_REL_AMD64_REL32),
  STRING_PAIR(IMAGE_REL_AMD64_REL32_1),
  STRING_PAIR(IMAGE_REL_AMD64_REL32_2),
  STRING_PAIR(IMAGE_REL_AMD64_REL32_3),
  STRING_PAIR(IMAGE_REL_AMD64_REL32_4),
  STRING_PAIR(IMAGE_REL_AMD64_REL32_5),
  STRING_PAIR(IMAGE_REL_AMD64_SECTION),
  STRING_PAIR(IMAGE_REL_AMD64_SECREL),
  STRING_PAIR(IMAGE_REL_AMD64_SECREL7),
  STRING_PAIR(IMAGE_REL_AMD64_TOKEN),
  STRING_PAIR(IMAGE_REL_AMD64_SREL32),
  STRING_PAIR(IMAGE_REL_AMD64_PAIR),
  STRING_PAIR(IMAGE_REL_AMD64_SSPAN32)
};

static const pod_pair<llvm::COFF::RelocationTypesARM, const char *> 
RelocationTypesARMPairs [] = {
  STRING_PAIR(IMAGE_REL_ARM_ABSOLUTE),
  STRING_PAIR(IMAGE_REL_ARM_ADDR32),
  STRING_PAIR(IMAGE_REL_ARM_ADDR32NB),
  STRING_PAIR(IMAGE_REL_ARM_BRANCH24),
  STRING_PAIR(IMAGE_REL_ARM_BRANCH11),
  STRING_PAIR(IMAGE_REL_ARM_TOKEN),
  STRING_PAIR(IMAGE_REL_ARM_BLX24),
  STRING_PAIR(IMAGE_REL_ARM_BLX11),
  STRING_PAIR(IMAGE_REL_ARM_SECTION),
  STRING_PAIR(IMAGE_REL_ARM_SECREL),
  STRING_PAIR(IMAGE_REL_ARM_MOV32A),
  STRING_PAIR(IMAGE_REL_ARM_MOV32T),
  STRING_PAIR(IMAGE_REL_ARM_BRANCH20T),
  STRING_PAIR(IMAGE_REL_ARM_BRANCH24T),
  STRING_PAIR(IMAGE_REL_ARM_BLX23T)
};
#undef STRING_PAIR


static const char endl = '\n';

namespace yaml {  // COFF-specific yaml-writing specific routines

static llvm::raw_ostream &writeName(llvm::raw_ostream &Out, 
                             const char *Name, std::size_t NameSize) {
  for (std::size_t i = 0; i < NameSize; ++i) {
    if (!Name[i]) break;
    Out << Name[i];
  }
  return Out;
}

// Given an array of pod_pair<enum, const char *>, write all enums that match
template <typename T, std::size_t N>
static llvm::raw_ostream &writeBitMask(llvm::raw_ostream &Out, 
              const pod_pair<T, const char *> (&Arr)[N], unsigned long Val) {
  for (std::size_t i = 0; i < N; ++i)
    if (Val & Arr[i].first)
      Out << Arr[i].second << ", ";
  return Out;
}

} // end of yaml namespace

// Given an array of pod_pair<enum, const char *>, look up a value
template <typename T, std::size_t N>
const char *nameLookup(const pod_pair<T, const char *> (&Arr)[N], 
                           unsigned long Val, const char *NotFound = NULL) {
  T n = static_cast<T>(Val);
  for (std::size_t i = 0; i < N; ++i)
    if (n == Arr[i].first)
      return Arr[i].second;
  return NotFound;
}


static llvm::raw_ostream &yamlCOFFHeader(
          const llvm::object::coff_file_header *Header,llvm::raw_ostream &Out) {

  Out << "header: !Header" << endl;
  Out << "  Machine: ";
  Out << nameLookup(MachineTypePairs, Header->Machine, "# Unknown_MachineTypes")
      << " # (";
  return yaml::writeHexNumber(Out, Header->Machine) << ")" << endl << endl;
}


static llvm::raw_ostream &yamlCOFFSections(llvm::object::COFFObjectFile &Obj, 
                            std::size_t NumSections, llvm::raw_ostream &Out) {
  llvm::error_code ec;
  Out << "sections:" << endl;
  for (llvm::object::section_iterator iter = Obj.begin_sections(); 
                             iter != Obj.end_sections(); iter.increment(ec)) {
    const llvm::object::coff_section *sect = Obj.getCOFFSection(iter);
  
    Out << "  - !Section" << endl;
    Out << "    Name: ";
    yaml::writeName(Out, sect->Name, sizeof(sect->Name)) << endl;

    Out << "    Characteristics: [";
    yaml::writeBitMask(Out, SectionCharacteristicsPairs1, sect->Characteristics);
    Out << nameLookup(SectionCharacteristicsPairsAlignment, 
        sect->Characteristics & 0x00F00000, "# Unrecognized_IMAGE_SCN_ALIGN") 
        << ", ";
    yaml::writeBitMask(Out, SectionCharacteristicsPairs2, sect->Characteristics);
    Out << "] # ";
    yaml::writeHexNumber(Out, sect->Characteristics) << endl;

    llvm::ArrayRef<uint8_t> sectionData;
    Obj.getSectionContents(sect, sectionData);    
    Out << "    SectionData: ";
    yaml::writeHexStream(Out, sectionData) << endl;
    if (iter->begin_relocations() != iter->end_relocations())
      Out << "    Relocations:\n";
    for (llvm::object::relocation_iterator rIter = iter->begin_relocations();
                       rIter != iter->end_relocations(); rIter.increment(ec)) {
      const llvm::object::coff_relocation *reloc = Obj.getCOFFRelocation(rIter);

        Out << "      - !Relocation" << endl;
        Out << "        VirtualAddress: " ;
        yaml::writeHexNumber(Out, reloc->VirtualAddress) << endl;
        Out << "        SymbolTableIndex: " << reloc->SymbolTableIndex << endl;
        Out << "        Type: " 
            << nameLookup(RelocationTypeX86Pairs, reloc->Type) << endl;
    // TODO: Use the correct reloc type for the machine.
        Out << endl;
      }

  } 
  return Out;
}

static llvm::raw_ostream& yamlCOFFSymbols(llvm::object::COFFObjectFile &Obj, 
                              std::size_t NumSymbols, llvm::raw_ostream &Out) {
  llvm::error_code ec;
  Out << "symbols:" << endl;
  for (llvm::object::symbol_iterator iter = Obj.begin_symbols(); 
                             iter != Obj.end_symbols(); iter.increment(ec)) {
 // Gather all the info that we need
    llvm::StringRef str;
    const llvm::object::coff_symbol *symbol = Obj.getCOFFSymbol(iter);
    Obj.getSymbolName(symbol, str);
    std::size_t  simpleType  = symbol->getBaseType();
    std::size_t complexType  = symbol->getComplexType();
    std::size_t storageClass = symbol->StorageClass;
    
    Out << "  - !Symbol" << endl;
    Out << "    Name: " << str << endl; 

    Out << "    Value: "         << symbol->Value << endl;
    Out << "    SectionNumber: " << symbol->SectionNumber << endl;

    Out << "    SimpleType: " 
        << nameLookup(SymbolBaseTypePairs, simpleType, 
            "# Unknown_SymbolBaseType") 
        << " # (" << simpleType << ")" << endl;
    
    Out << "    ComplexType: " 
        << nameLookup(SymbolComplexTypePairs, complexType, 
                "# Unknown_SymbolComplexType") 
        << " # (" << complexType << ")" << endl;
    
    Out << "    StorageClass: " 
        << nameLookup(SymbolStorageClassPairs, storageClass,
              "# Unknown_StorageClass") 
        << " # (" << (int) storageClass << ")" << endl;

    if (symbol->NumberOfAuxSymbols > 0) {
      llvm::ArrayRef<uint8_t> aux = Obj.getSymbolAuxData(symbol);
      Out << "    NumberOfAuxSymbols: " 
          << (int) symbol->NumberOfAuxSymbols << endl;
      Out << "    AuxillaryData: ";
      yaml::writeHexStream(Out, aux);
    }
      
    Out << endl;
  }

  return Out;
}


llvm::error_code coff2yaml(llvm::raw_ostream &Out, llvm::MemoryBuffer *TheObj) {
  llvm::error_code ec;
  llvm::object::COFFObjectFile obj(TheObj, ec);
  if (!ec) {
    const llvm::object::coff_file_header *hd;
    ec = obj.getHeader(hd);
    if (!ec) {
      yamlCOFFHeader(hd, Out);
      yamlCOFFSections(obj, hd->NumberOfSections, Out);
      yamlCOFFSymbols(obj, hd->NumberOfSymbols, Out);
    }
  }
  return ec;
}
