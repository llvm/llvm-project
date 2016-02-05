//===-- COFFDumper.cpp - COFF-specific dumper -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements the COFF-specific dumper for llvm-readobj.
///
//===----------------------------------------------------------------------===//

#include "llvm-readobj.h"
#include "ARMWinEHPrinter.h"
#include "CodeView.h"
#include "Error.h"
#include "ObjDumper.h"
#include "StackMapPrinter.h"
#include "StreamWriter.h"
#include "Win64EHDumper.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/Line.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Win64EH.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstring>
#include <system_error>
#include <time.h>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::codeview;
using namespace llvm::support;
using namespace llvm::Win64EH;

namespace {

class COFFDumper : public ObjDumper {
public:
  COFFDumper(const llvm::object::COFFObjectFile *Obj, StreamWriter& Writer)
    : ObjDumper(Writer)
    , Obj(Obj) {
  }

  void printFileHeaders() override;
  void printSections() override;
  void printRelocations() override;
  void printSymbols() override;
  void printDynamicSymbols() override;
  void printUnwindInfo() override;
  void printCOFFImports() override;
  void printCOFFExports() override;
  void printCOFFDirectives() override;
  void printCOFFBaseReloc() override;
  void printCodeViewDebugInfo() override;
  void printStackMap() const override;
private:
  void printSymbol(const SymbolRef &Sym);
  void printRelocation(const SectionRef &Section, const RelocationRef &Reloc,
                       uint64_t Bias = 0);
  void printDataDirectory(uint32_t Index, const std::string &FieldName);

  void printDOSHeader(const dos_header *DH);
  template <class PEHeader> void printPEHeader(const PEHeader *Hdr);
  void printBaseOfDataField(const pe32_header *Hdr);
  void printBaseOfDataField(const pe32plus_header *Hdr);

  void printCodeViewSymbolSection(StringRef SectionName, const SectionRef &Section);
  void printCodeViewTypeSection(StringRef SectionName, const SectionRef &Section);
  void printCodeViewFieldList(StringRef FieldData);
  StringRef getTypeName(TypeIndex Ty);
  StringRef getFileNameForFileOffset(uint32_t FileOffset);
  void printFileNameForOffset(StringRef Label, uint32_t FileOffset);
  void printTypeIndex(StringRef FieldName, TypeIndex TI);
  void printLocalVariableAddrRange(const LocalVariableAddrRange &Range,
                                   const coff_section *Sec,
                                   StringRef SectionContents);
  void printLocalVariableAddrGap(StringRef &SymData);

  void printCodeViewSymbolsSubsection(StringRef Subsection,
                                      const SectionRef &Section,
                                      StringRef SectionContents);

  void printCodeViewFileChecksums(StringRef Subsection);

  void printCodeViewInlineeLines(StringRef Subsection);

  void printMemberAttributes(MemberAttributes Attrs);

  void printRelocatedField(StringRef Label, const coff_section *Sec,
                           StringRef SectionContents, const ulittle32_t *Field,
                           StringRef *RelocSym = nullptr);

  void printBinaryBlockWithRelocs(StringRef Label, const SectionRef &Sec,
                                  StringRef SectionContents, StringRef Block);

  /// Given a .debug$S section, find the string table and file checksum table.
  void initializeFileAndStringTables(StringRef Data);

  void cacheRelocations();

  std::error_code resolveSymbol(const coff_section *Section, uint64_t Offset,
                                SymbolRef &Sym);
  std::error_code resolveSymbolName(const coff_section *Section,
                                    uint64_t Offset, StringRef &Name);
  std::error_code resolveSymbolName(const coff_section *Section,
                                    StringRef SectionContents,
                                    const void *RelocPtr, StringRef &Name);
  void printImportedSymbols(iterator_range<imported_symbol_iterator> Range);
  void printDelayImportedSymbols(
      const DelayImportDirectoryEntryRef &I,
      iterator_range<imported_symbol_iterator> Range);

  typedef DenseMap<const coff_section*, std::vector<RelocationRef> > RelocMapTy;

  const llvm::object::COFFObjectFile *Obj;
  bool RelocCached = false;
  RelocMapTy RelocMap;
  StringRef CVFileChecksumTable;
  StringRef CVStringTable;

  /// All user defined type records in .debug$T live in here. Type indices
  /// greater than 0x1000 are user defined. Subtract 0x1000 from the index to
  /// index into this vector.
  SmallVector<StringRef, 10> CVUDTNames;

  StringSet<> TypeNames;
};

} // namespace


namespace llvm {

std::error_code createCOFFDumper(const object::ObjectFile *Obj,
                                 StreamWriter &Writer,
                                 std::unique_ptr<ObjDumper> &Result) {
  const COFFObjectFile *COFFObj = dyn_cast<COFFObjectFile>(Obj);
  if (!COFFObj)
    return readobj_error::unsupported_obj_file_format;

  Result.reset(new COFFDumper(COFFObj, Writer));
  return readobj_error::success;
}

} // namespace llvm

// Given a a section and an offset into this section the function returns the
// symbol used for the relocation at the offset.
std::error_code COFFDumper::resolveSymbol(const coff_section *Section,
                                          uint64_t Offset, SymbolRef &Sym) {
  cacheRelocations();
  const auto &Relocations = RelocMap[Section];
  for (const auto &Relocation : Relocations) {
    uint64_t RelocationOffset = Relocation.getOffset();

    if (RelocationOffset == Offset) {
      Sym = *Relocation.getSymbol();
      return readobj_error::success;
    }
  }
  return readobj_error::unknown_symbol;
}

// Given a section and an offset into this section the function returns the name
// of the symbol used for the relocation at the offset.
std::error_code COFFDumper::resolveSymbolName(const coff_section *Section,
                                              uint64_t Offset,
                                              StringRef &Name) {
  SymbolRef Symbol;
  if (std::error_code EC = resolveSymbol(Section, Offset, Symbol))
    return EC;
  ErrorOr<StringRef> NameOrErr = Symbol.getName();
  if (std::error_code EC = NameOrErr.getError())
    return EC;
  Name = *NameOrErr;
  return std::error_code();
}

// Helper for when you have a pointer to real data and you want to know about
// relocations against it.
std::error_code COFFDumper::resolveSymbolName(const coff_section *Section,
                                              StringRef SectionContents,
                                              const void *RelocPtr,
                                              StringRef &Name) {
  assert(SectionContents.data() < RelocPtr &&
         RelocPtr < SectionContents.data() + SectionContents.size() &&
         "pointer to relocated object is not in section");
  uint64_t Offset = ptrdiff_t(reinterpret_cast<const char *>(RelocPtr) -
                              SectionContents.data());
  return resolveSymbolName(Section, Offset, Name);
}

void COFFDumper::printRelocatedField(StringRef Label, const coff_section *Sec,
                                     StringRef SectionContents,
                                     const ulittle32_t *Field,
                                     StringRef *RelocSym) {
  StringRef SymStorage;
  StringRef &Symbol = RelocSym ? *RelocSym : SymStorage;
  if (!resolveSymbolName(Sec, SectionContents, Field, Symbol))
    W.printSymbolOffset(Label, Symbol, *Field);
  else
    W.printHex(Label, *Field);
}

void COFFDumper::printBinaryBlockWithRelocs(StringRef Label,
                                            const SectionRef &Sec,
                                            StringRef SectionContents,
                                            StringRef Block) {
  W.printBinaryBlock(Label, Block);

  assert(SectionContents.begin() < Block.begin() &&
         SectionContents.end() >= Block.end() &&
         "Block is not contained in SectionContents");
  uint64_t OffsetStart = Block.data() - SectionContents.data();
  uint64_t OffsetEnd = OffsetStart + Block.size();

  cacheRelocations();
  ListScope D(W, "BlockRelocations");
  const coff_section *Section = Obj->getCOFFSection(Sec);
  const auto &Relocations = RelocMap[Section];
  for (const auto &Relocation : Relocations) {
    uint64_t RelocationOffset = Relocation.getOffset();
    if (OffsetStart <= RelocationOffset && RelocationOffset < OffsetEnd)
      printRelocation(Sec, Relocation, OffsetStart);
  }
}

static const EnumEntry<COFF::MachineTypes> ImageFileMachineType[] = {
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_UNKNOWN  ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_AM33     ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_AMD64    ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_ARM      ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_ARMNT    ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_EBC      ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_I386     ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_IA64     ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_M32R     ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_MIPS16   ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_MIPSFPU  ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_MIPSFPU16),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_POWERPC  ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_POWERPCFP),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_R4000    ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_SH3      ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_SH3DSP   ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_SH4      ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_SH5      ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_THUMB    ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_MACHINE_WCEMIPSV2)
};

static const EnumEntry<COFF::Characteristics> ImageFileCharacteristics[] = {
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_RELOCS_STRIPPED        ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_EXECUTABLE_IMAGE       ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_LINE_NUMS_STRIPPED     ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_LOCAL_SYMS_STRIPPED    ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_AGGRESSIVE_WS_TRIM     ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_LARGE_ADDRESS_AWARE    ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_BYTES_REVERSED_LO      ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_32BIT_MACHINE          ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_DEBUG_STRIPPED         ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_REMOVABLE_RUN_FROM_SWAP),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_NET_RUN_FROM_SWAP      ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_SYSTEM                 ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_DLL                    ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_UP_SYSTEM_ONLY         ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_FILE_BYTES_REVERSED_HI      )
};

static const EnumEntry<COFF::WindowsSubsystem> PEWindowsSubsystem[] = {
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SUBSYSTEM_UNKNOWN                ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SUBSYSTEM_NATIVE                 ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SUBSYSTEM_WINDOWS_GUI            ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SUBSYSTEM_WINDOWS_CUI            ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SUBSYSTEM_POSIX_CUI              ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SUBSYSTEM_WINDOWS_CE_GUI         ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SUBSYSTEM_EFI_APPLICATION        ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SUBSYSTEM_EFI_BOOT_SERVICE_DRIVER),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SUBSYSTEM_EFI_RUNTIME_DRIVER     ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SUBSYSTEM_EFI_ROM                ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SUBSYSTEM_XBOX                   ),
};

static const EnumEntry<COFF::DLLCharacteristics> PEDLLCharacteristics[] = {
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_DLL_CHARACTERISTICS_HIGH_ENTROPY_VA      ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_DLL_CHARACTERISTICS_DYNAMIC_BASE         ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_DLL_CHARACTERISTICS_FORCE_INTEGRITY      ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_DLL_CHARACTERISTICS_NX_COMPAT            ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_DLL_CHARACTERISTICS_NO_ISOLATION         ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_DLL_CHARACTERISTICS_NO_SEH               ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_DLL_CHARACTERISTICS_NO_BIND              ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_DLL_CHARACTERISTICS_APPCONTAINER         ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_DLL_CHARACTERISTICS_WDM_DRIVER           ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_DLL_CHARACTERISTICS_GUARD_CF             ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_DLL_CHARACTERISTICS_TERMINAL_SERVER_AWARE),
};

static const EnumEntry<COFF::SectionCharacteristics>
ImageSectionCharacteristics[] = {
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_TYPE_NOLOAD           ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_TYPE_NO_PAD           ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_CNT_CODE              ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_CNT_INITIALIZED_DATA  ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_CNT_UNINITIALIZED_DATA),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_LNK_OTHER             ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_LNK_INFO              ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_LNK_REMOVE            ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_LNK_COMDAT            ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_GPREL                 ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_MEM_PURGEABLE         ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_MEM_16BIT             ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_MEM_LOCKED            ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_MEM_PRELOAD           ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_1BYTES          ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_2BYTES          ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_4BYTES          ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_8BYTES          ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_16BYTES         ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_32BYTES         ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_64BYTES         ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_128BYTES        ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_256BYTES        ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_512BYTES        ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_1024BYTES       ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_2048BYTES       ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_4096BYTES       ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_8192BYTES       ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_LNK_NRELOC_OVFL       ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_MEM_DISCARDABLE       ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_MEM_NOT_CACHED        ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_MEM_NOT_PAGED         ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_MEM_SHARED            ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_MEM_EXECUTE           ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_MEM_READ              ),
  LLVM_READOBJ_ENUM_ENT(COFF, IMAGE_SCN_MEM_WRITE             )
};

static const EnumEntry<COFF::SymbolBaseType> ImageSymType[] = {
  { "Null"  , COFF::IMAGE_SYM_TYPE_NULL   },
  { "Void"  , COFF::IMAGE_SYM_TYPE_VOID   },
  { "Char"  , COFF::IMAGE_SYM_TYPE_CHAR   },
  { "Short" , COFF::IMAGE_SYM_TYPE_SHORT  },
  { "Int"   , COFF::IMAGE_SYM_TYPE_INT    },
  { "Long"  , COFF::IMAGE_SYM_TYPE_LONG   },
  { "Float" , COFF::IMAGE_SYM_TYPE_FLOAT  },
  { "Double", COFF::IMAGE_SYM_TYPE_DOUBLE },
  { "Struct", COFF::IMAGE_SYM_TYPE_STRUCT },
  { "Union" , COFF::IMAGE_SYM_TYPE_UNION  },
  { "Enum"  , COFF::IMAGE_SYM_TYPE_ENUM   },
  { "MOE"   , COFF::IMAGE_SYM_TYPE_MOE    },
  { "Byte"  , COFF::IMAGE_SYM_TYPE_BYTE   },
  { "Word"  , COFF::IMAGE_SYM_TYPE_WORD   },
  { "UInt"  , COFF::IMAGE_SYM_TYPE_UINT   },
  { "DWord" , COFF::IMAGE_SYM_TYPE_DWORD  }
};

static const EnumEntry<COFF::SymbolComplexType> ImageSymDType[] = {
  { "Null"    , COFF::IMAGE_SYM_DTYPE_NULL     },
  { "Pointer" , COFF::IMAGE_SYM_DTYPE_POINTER  },
  { "Function", COFF::IMAGE_SYM_DTYPE_FUNCTION },
  { "Array"   , COFF::IMAGE_SYM_DTYPE_ARRAY    }
};

static const EnumEntry<COFF::SymbolStorageClass> ImageSymClass[] = {
  { "EndOfFunction"  , COFF::IMAGE_SYM_CLASS_END_OF_FUNCTION  },
  { "Null"           , COFF::IMAGE_SYM_CLASS_NULL             },
  { "Automatic"      , COFF::IMAGE_SYM_CLASS_AUTOMATIC        },
  { "External"       , COFF::IMAGE_SYM_CLASS_EXTERNAL         },
  { "Static"         , COFF::IMAGE_SYM_CLASS_STATIC           },
  { "Register"       , COFF::IMAGE_SYM_CLASS_REGISTER         },
  { "ExternalDef"    , COFF::IMAGE_SYM_CLASS_EXTERNAL_DEF     },
  { "Label"          , COFF::IMAGE_SYM_CLASS_LABEL            },
  { "UndefinedLabel" , COFF::IMAGE_SYM_CLASS_UNDEFINED_LABEL  },
  { "MemberOfStruct" , COFF::IMAGE_SYM_CLASS_MEMBER_OF_STRUCT },
  { "Argument"       , COFF::IMAGE_SYM_CLASS_ARGUMENT         },
  { "StructTag"      , COFF::IMAGE_SYM_CLASS_STRUCT_TAG       },
  { "MemberOfUnion"  , COFF::IMAGE_SYM_CLASS_MEMBER_OF_UNION  },
  { "UnionTag"       , COFF::IMAGE_SYM_CLASS_UNION_TAG        },
  { "TypeDefinition" , COFF::IMAGE_SYM_CLASS_TYPE_DEFINITION  },
  { "UndefinedStatic", COFF::IMAGE_SYM_CLASS_UNDEFINED_STATIC },
  { "EnumTag"        , COFF::IMAGE_SYM_CLASS_ENUM_TAG         },
  { "MemberOfEnum"   , COFF::IMAGE_SYM_CLASS_MEMBER_OF_ENUM   },
  { "RegisterParam"  , COFF::IMAGE_SYM_CLASS_REGISTER_PARAM   },
  { "BitField"       , COFF::IMAGE_SYM_CLASS_BIT_FIELD        },
  { "Block"          , COFF::IMAGE_SYM_CLASS_BLOCK            },
  { "Function"       , COFF::IMAGE_SYM_CLASS_FUNCTION         },
  { "EndOfStruct"    , COFF::IMAGE_SYM_CLASS_END_OF_STRUCT    },
  { "File"           , COFF::IMAGE_SYM_CLASS_FILE             },
  { "Section"        , COFF::IMAGE_SYM_CLASS_SECTION          },
  { "WeakExternal"   , COFF::IMAGE_SYM_CLASS_WEAK_EXTERNAL    },
  { "CLRToken"       , COFF::IMAGE_SYM_CLASS_CLR_TOKEN        }
};

static const EnumEntry<COFF::COMDATType> ImageCOMDATSelect[] = {
  { "NoDuplicates", COFF::IMAGE_COMDAT_SELECT_NODUPLICATES },
  { "Any"         , COFF::IMAGE_COMDAT_SELECT_ANY          },
  { "SameSize"    , COFF::IMAGE_COMDAT_SELECT_SAME_SIZE    },
  { "ExactMatch"  , COFF::IMAGE_COMDAT_SELECT_EXACT_MATCH  },
  { "Associative" , COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE  },
  { "Largest"     , COFF::IMAGE_COMDAT_SELECT_LARGEST      },
  { "Newest"      , COFF::IMAGE_COMDAT_SELECT_NEWEST       }
};

static const EnumEntry<COFF::WeakExternalCharacteristics>
WeakExternalCharacteristics[] = {
  { "NoLibrary", COFF::IMAGE_WEAK_EXTERN_SEARCH_NOLIBRARY },
  { "Library"  , COFF::IMAGE_WEAK_EXTERN_SEARCH_LIBRARY   },
  { "Alias"    , COFF::IMAGE_WEAK_EXTERN_SEARCH_ALIAS     }
};

static const EnumEntry<CompileSym3::Flags> CompileSym3Flags[] = {
    LLVM_READOBJ_ENUM_ENT(CompileSym3, EC),
    LLVM_READOBJ_ENUM_ENT(CompileSym3, NoDbgInfo),
    LLVM_READOBJ_ENUM_ENT(CompileSym3, LTCG),
    LLVM_READOBJ_ENUM_ENT(CompileSym3, NoDataAlign),
    LLVM_READOBJ_ENUM_ENT(CompileSym3, ManagedPresent),
    LLVM_READOBJ_ENUM_ENT(CompileSym3, SecurityChecks),
    LLVM_READOBJ_ENUM_ENT(CompileSym3, HotPatch),
    LLVM_READOBJ_ENUM_ENT(CompileSym3, CVTCIL),
    LLVM_READOBJ_ENUM_ENT(CompileSym3, MSILModule),
    LLVM_READOBJ_ENUM_ENT(CompileSym3, Sdl),
    LLVM_READOBJ_ENUM_ENT(CompileSym3, PGO),
    LLVM_READOBJ_ENUM_ENT(CompileSym3, Exp),
};

static const EnumEntry<codeview::SourceLanguage> SourceLanguages[] = {
    LLVM_READOBJ_ENUM_ENT(SourceLanguage, C),
    LLVM_READOBJ_ENUM_ENT(SourceLanguage, Cpp),
    LLVM_READOBJ_ENUM_ENT(SourceLanguage, Fortran),
    LLVM_READOBJ_ENUM_ENT(SourceLanguage, Masm),
    LLVM_READOBJ_ENUM_ENT(SourceLanguage, Pascal),
    LLVM_READOBJ_ENUM_ENT(SourceLanguage, Basic),
    LLVM_READOBJ_ENUM_ENT(SourceLanguage, Cobol),
    LLVM_READOBJ_ENUM_ENT(SourceLanguage, Link),
    LLVM_READOBJ_ENUM_ENT(SourceLanguage, Cvtres),
    LLVM_READOBJ_ENUM_ENT(SourceLanguage, Cvtpgd),
    LLVM_READOBJ_ENUM_ENT(SourceLanguage, CSharp),
    LLVM_READOBJ_ENUM_ENT(SourceLanguage, VB),
    LLVM_READOBJ_ENUM_ENT(SourceLanguage, ILAsm),
    LLVM_READOBJ_ENUM_ENT(SourceLanguage, Java),
    LLVM_READOBJ_ENUM_ENT(SourceLanguage, JScript),
    LLVM_READOBJ_ENUM_ENT(SourceLanguage, MSIL),
    LLVM_READOBJ_ENUM_ENT(SourceLanguage, HLSL),
};

static const EnumEntry<uint32_t> SubSectionTypes[] = {
  LLVM_READOBJ_ENUM_CLASS_ENT(ModuleSubstreamKind, Symbols),
  LLVM_READOBJ_ENUM_CLASS_ENT(ModuleSubstreamKind, Lines),
  LLVM_READOBJ_ENUM_CLASS_ENT(ModuleSubstreamKind, StringTable),
  LLVM_READOBJ_ENUM_CLASS_ENT(ModuleSubstreamKind, FileChecksums),
  LLVM_READOBJ_ENUM_CLASS_ENT(ModuleSubstreamKind, FrameData),
  LLVM_READOBJ_ENUM_CLASS_ENT(ModuleSubstreamKind, InlineeLines),
  LLVM_READOBJ_ENUM_CLASS_ENT(ModuleSubstreamKind, CrossScopeImports),
  LLVM_READOBJ_ENUM_CLASS_ENT(ModuleSubstreamKind, CrossScopeExports),
  LLVM_READOBJ_ENUM_CLASS_ENT(ModuleSubstreamKind, ILLines),
  LLVM_READOBJ_ENUM_CLASS_ENT(ModuleSubstreamKind, FuncMDTokenMap),
  LLVM_READOBJ_ENUM_CLASS_ENT(ModuleSubstreamKind, TypeMDTokenMap),
  LLVM_READOBJ_ENUM_CLASS_ENT(ModuleSubstreamKind, MergedAssemblyInput),
  LLVM_READOBJ_ENUM_CLASS_ENT(ModuleSubstreamKind, CoffSymbolRVA),
};

static const EnumEntry<unsigned> CPUTypeNames[] = {
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, Intel8080),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, Intel8086),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, Intel80286),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, Intel80386),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, Intel80486),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, Pentium),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, PentiumPro),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, Pentium3),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, MIPS),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, MIPS16),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, MIPS32),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, MIPS64),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, MIPSI),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, MIPSII),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, MIPSIII),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, MIPSIV),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, MIPSV),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, M68000),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, M68010),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, M68020),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, M68030),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, M68040),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, Alpha),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, Alpha21164),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, Alpha21164A),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, Alpha21264),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, Alpha21364),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, PPC601),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, PPC603),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, PPC604),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, PPC620),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, PPCFP),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, PPCBE),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, SH3),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, SH3E),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, SH3DSP),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, SH4),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, SHMedia),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, ARM3),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, ARM4),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, ARM4T),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, ARM5),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, ARM5T),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, ARM6),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, ARM_XMAC),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, ARM_WMMX),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, ARM7),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, Omni),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, Ia64),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, Ia64_2),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, CEE),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, AM33),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, M32R),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, TriCore),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, X64),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, EBC),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, Thumb),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, ARMNT),
  LLVM_READOBJ_ENUM_CLASS_ENT(CPUType, D3D11_Shader),
};

static const EnumEntry<uint8_t> ProcSymFlags[] = {
    LLVM_READOBJ_ENUM_ENT(ProcFlags, HasFP),
    LLVM_READOBJ_ENUM_ENT(ProcFlags, HasIRET),
    LLVM_READOBJ_ENUM_ENT(ProcFlags, HasFRET),
    LLVM_READOBJ_ENUM_ENT(ProcFlags, IsNoReturn),
    LLVM_READOBJ_ENUM_ENT(ProcFlags, IsUnreachable),
    LLVM_READOBJ_ENUM_ENT(ProcFlags, HasCustomCallingConv),
    LLVM_READOBJ_ENUM_ENT(ProcFlags, IsNoInline),
    LLVM_READOBJ_ENUM_ENT(ProcFlags, HasOptimizedDebugInfo),
};

static const EnumEntry<uint32_t> FrameProcSymFlags[] = {
    LLVM_READOBJ_ENUM_CLASS_ENT(FrameProcedureOptions, HasAlloca),
    LLVM_READOBJ_ENUM_CLASS_ENT(FrameProcedureOptions, HasSetJmp),
    LLVM_READOBJ_ENUM_CLASS_ENT(FrameProcedureOptions, HasLongJmp),
    LLVM_READOBJ_ENUM_CLASS_ENT(FrameProcedureOptions, HasInlineAssembly),
    LLVM_READOBJ_ENUM_CLASS_ENT(FrameProcedureOptions, HasExceptionHandling),
    LLVM_READOBJ_ENUM_CLASS_ENT(FrameProcedureOptions, MarkedInline),
    LLVM_READOBJ_ENUM_CLASS_ENT(FrameProcedureOptions,
                                HasStructuredExceptionHandling),
    LLVM_READOBJ_ENUM_CLASS_ENT(FrameProcedureOptions, Naked),
    LLVM_READOBJ_ENUM_CLASS_ENT(FrameProcedureOptions, SecurityChecks),
    LLVM_READOBJ_ENUM_CLASS_ENT(FrameProcedureOptions,
                                AsynchronousExceptionHandling),
    LLVM_READOBJ_ENUM_CLASS_ENT(FrameProcedureOptions,
                                NoStackOrderingForSecurityChecks),
    LLVM_READOBJ_ENUM_CLASS_ENT(FrameProcedureOptions, Inlined),
    LLVM_READOBJ_ENUM_CLASS_ENT(FrameProcedureOptions, StrictSecurityChecks),
    LLVM_READOBJ_ENUM_CLASS_ENT(FrameProcedureOptions, SafeBuffers),
    LLVM_READOBJ_ENUM_CLASS_ENT(FrameProcedureOptions,
                                ProfileGuidedOptimization),
    LLVM_READOBJ_ENUM_CLASS_ENT(FrameProcedureOptions, ValidProfileCounts),
    LLVM_READOBJ_ENUM_CLASS_ENT(FrameProcedureOptions, OptimizedForSpeed),
    LLVM_READOBJ_ENUM_CLASS_ENT(FrameProcedureOptions, GuardCfg),
    LLVM_READOBJ_ENUM_CLASS_ENT(FrameProcedureOptions, GuardCfw),
};

static const EnumEntry<uint32_t> FrameDataFlags[] = {
    LLVM_READOBJ_ENUM_ENT(FrameData, HasSEH),
    LLVM_READOBJ_ENUM_ENT(FrameData, HasEH),
    LLVM_READOBJ_ENUM_ENT(FrameData, IsFunctionStart),
};

static const EnumEntry<uint16_t> LocalFlags[] = {
    LLVM_READOBJ_ENUM_ENT(LocalSym, IsParameter),
    LLVM_READOBJ_ENUM_ENT(LocalSym, IsAddressTaken),
    LLVM_READOBJ_ENUM_ENT(LocalSym, IsCompilerGenerated),
    LLVM_READOBJ_ENUM_ENT(LocalSym, IsAggregate),
    LLVM_READOBJ_ENUM_ENT(LocalSym, IsAggregated),
    LLVM_READOBJ_ENUM_ENT(LocalSym, IsAliased),
    LLVM_READOBJ_ENUM_ENT(LocalSym, IsAlias),
    LLVM_READOBJ_ENUM_ENT(LocalSym, IsReturnValue),
    LLVM_READOBJ_ENUM_ENT(LocalSym, IsOptimizedOut),
    LLVM_READOBJ_ENUM_ENT(LocalSym, IsEnregisteredGlobal),
    LLVM_READOBJ_ENUM_ENT(LocalSym, IsEnregisteredStatic),
};

static const EnumEntry<uint16_t> FrameCookieKinds[] = {
    LLVM_READOBJ_ENUM_ENT(FrameCookieSym, Copy),
    LLVM_READOBJ_ENUM_ENT(FrameCookieSym, XorStackPointer),
    LLVM_READOBJ_ENUM_ENT(FrameCookieSym, XorFramePointer),
    LLVM_READOBJ_ENUM_ENT(FrameCookieSym, XorR13),
};

static const EnumEntry<uint16_t> ClassOptionNames[] = {
  LLVM_READOBJ_ENUM_CLASS_ENT(ClassOptions, Packed),
  LLVM_READOBJ_ENUM_CLASS_ENT(ClassOptions, HasConstructorOrDestructor),
  LLVM_READOBJ_ENUM_CLASS_ENT(ClassOptions, HasOverloadedOperator),
  LLVM_READOBJ_ENUM_CLASS_ENT(ClassOptions, Nested),
  LLVM_READOBJ_ENUM_CLASS_ENT(ClassOptions, ContainsNestedClass),
  LLVM_READOBJ_ENUM_CLASS_ENT(ClassOptions, HasOverloadedAssignmentOperator),
  LLVM_READOBJ_ENUM_CLASS_ENT(ClassOptions, HasConversionOperator),
  LLVM_READOBJ_ENUM_CLASS_ENT(ClassOptions, ForwardReference),
  LLVM_READOBJ_ENUM_CLASS_ENT(ClassOptions, Scoped),
  LLVM_READOBJ_ENUM_CLASS_ENT(ClassOptions, HasUniqueName),
  LLVM_READOBJ_ENUM_CLASS_ENT(ClassOptions, Sealed),
  LLVM_READOBJ_ENUM_CLASS_ENT(ClassOptions, Intrinsic),
};

static const EnumEntry<uint8_t> MemberAccessNames[] = {
    LLVM_READOBJ_ENUM_CLASS_ENT(MemberAccess, None),
    LLVM_READOBJ_ENUM_CLASS_ENT(MemberAccess, Private),
    LLVM_READOBJ_ENUM_CLASS_ENT(MemberAccess, Protected),
    LLVM_READOBJ_ENUM_CLASS_ENT(MemberAccess, Public),
};

static const EnumEntry<uint16_t> MethodOptionNames[] = {
    LLVM_READOBJ_ENUM_CLASS_ENT(MethodOptions, Pseudo),
    LLVM_READOBJ_ENUM_CLASS_ENT(MethodOptions, NoInherit),
    LLVM_READOBJ_ENUM_CLASS_ENT(MethodOptions, NoConstruct),
    LLVM_READOBJ_ENUM_CLASS_ENT(MethodOptions, CompilerGenerated),
    LLVM_READOBJ_ENUM_CLASS_ENT(MethodOptions, Sealed),
};

static const EnumEntry<uint16_t> MemberKindNames[] = {
    LLVM_READOBJ_ENUM_CLASS_ENT(MethodKind, Vanilla),
    LLVM_READOBJ_ENUM_CLASS_ENT(MethodKind, Virtual),
    LLVM_READOBJ_ENUM_CLASS_ENT(MethodKind, Static),
    LLVM_READOBJ_ENUM_CLASS_ENT(MethodKind, Friend),
    LLVM_READOBJ_ENUM_CLASS_ENT(MethodKind, IntroducingVirtual),
    LLVM_READOBJ_ENUM_CLASS_ENT(MethodKind, PureVirtual),
    LLVM_READOBJ_ENUM_CLASS_ENT(MethodKind, PureIntroducingVirtual),
};

/// The names here all end in "*". If the simple type is a pointer type, we
/// return the whole name. Otherwise we lop off the last character in our
/// StringRef.
static const EnumEntry<SimpleTypeKind> SimpleTypeNames[] = {
    {"void*", SimpleTypeKind::Void},
    {"<not translated>*", SimpleTypeKind::NotTranslated},
    {"HRESULT*", SimpleTypeKind::HResult},
    {"signed char*", SimpleTypeKind::SignedCharacter},
    {"unsigned char*", SimpleTypeKind::UnsignedCharacter},
    {"char*", SimpleTypeKind::NarrowCharacter},
    {"wchar_t*", SimpleTypeKind::WideCharacter},
    {"__int8*", SimpleTypeKind::SByte},
    {"unsigned __int8*", SimpleTypeKind::Byte},
    {"short*", SimpleTypeKind::Int16Short},
    {"unsigned short*", SimpleTypeKind::UInt16Short},
    {"__int16*", SimpleTypeKind::Int16},
    {"unsigned __int16*", SimpleTypeKind::UInt16},
    {"long*", SimpleTypeKind::Int32Long},
    {"unsigned long*", SimpleTypeKind::UInt32Long},
    {"int*", SimpleTypeKind::Int32},
    {"unsigned*", SimpleTypeKind::UInt32},
    {"__int64*", SimpleTypeKind::Int64Quad},
    {"unsigned __int64*", SimpleTypeKind::UInt64Quad},
    {"__int64*", SimpleTypeKind::Int64},
    {"unsigned __int64*", SimpleTypeKind::UInt64},
    {"__int128*", SimpleTypeKind::Int128},
    {"unsigned __int128*", SimpleTypeKind::UInt128},
    {"__half*", SimpleTypeKind::Float16},
    {"float*", SimpleTypeKind::Float32},
    {"float*", SimpleTypeKind::Float32PartialPrecision},
    {"__float48*", SimpleTypeKind::Float48},
    {"double*", SimpleTypeKind::Float64},
    {"long double*", SimpleTypeKind::Float80},
    {"__float128*", SimpleTypeKind::Float128},
    {"_Complex float*", SimpleTypeKind::Complex32},
    {"_Complex double*", SimpleTypeKind::Complex64},
    {"_Complex long double*", SimpleTypeKind::Complex80},
    {"_Complex __float128*", SimpleTypeKind::Complex128},
    {"bool*", SimpleTypeKind::Boolean8},
    {"__bool16*", SimpleTypeKind::Boolean16},
    {"__bool32*", SimpleTypeKind::Boolean32},
    {"__bool64*", SimpleTypeKind::Boolean64},
};

static const EnumEntry<TypeLeafKind> LeafTypeNames[] = {
#define LEAF_TYPE(name, val) LLVM_READOBJ_ENUM_ENT(TypeLeafKind, name),
#include "llvm/DebugInfo/CodeView/CVLeafTypes.def"
};

static const EnumEntry<uint8_t> PtrKindNames[] = {
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerKind, Near16),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerKind, Far16),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerKind, Huge16),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerKind, BasedOnSegment),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerKind, BasedOnValue),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerKind, BasedOnSegmentValue),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerKind, BasedOnAddress),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerKind, BasedOnSegmentAddress),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerKind, BasedOnType),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerKind, BasedOnSelf),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerKind, Near32),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerKind, Far32),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerKind, Near64),
};

static const EnumEntry<uint8_t> PtrModeNames[] = {
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerMode, Pointer),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerMode, LValueReference),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerMode, PointerToDataMember),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerMode, PointerToMemberFunction),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerMode, RValueReference),
};

static const EnumEntry<uint16_t> PtrMemberRepNames[] = {
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerToMemberRepresentation, Unknown),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerToMemberRepresentation,
                                SingleInheritanceData),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerToMemberRepresentation,
                                MultipleInheritanceData),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerToMemberRepresentation,
                                VirtualInheritanceData),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerToMemberRepresentation, GeneralData),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerToMemberRepresentation,
                                SingleInheritanceFunction),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerToMemberRepresentation,
                                MultipleInheritanceFunction),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerToMemberRepresentation,
                                VirtualInheritanceFunction),
    LLVM_READOBJ_ENUM_CLASS_ENT(PointerToMemberRepresentation, GeneralFunction),
};

static const EnumEntry<uint16_t> TypeModifierNames[] = {
    LLVM_READOBJ_ENUM_CLASS_ENT(ModifierOptions, Const),
    LLVM_READOBJ_ENUM_CLASS_ENT(ModifierOptions, Volatile),
    LLVM_READOBJ_ENUM_CLASS_ENT(ModifierOptions, Unaligned),
};

static const EnumEntry<uint8_t> CallingConventions[] = {
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, NearC),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, FarC),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, NearPascal),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, FarPascal),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, NearFast),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, FarFast),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, NearStdCall),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, FarStdCall),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, NearSysCall),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, FarSysCall),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, ThisCall),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, MipsCall),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, Generic),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, AlphaCall),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, PpcCall),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, SHCall),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, ArmCall),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, AM33Call),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, TriCall),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, SH5Call),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, M32RCall),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, ClrCall),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, Inline),
    LLVM_READOBJ_ENUM_CLASS_ENT(CallingConvention, NearVector),
};

static const EnumEntry<uint8_t> FunctionOptionEnum[] = {
    LLVM_READOBJ_ENUM_CLASS_ENT(FunctionOptions, CxxReturnUdt),
    LLVM_READOBJ_ENUM_CLASS_ENT(FunctionOptions, Constructor),
    LLVM_READOBJ_ENUM_CLASS_ENT(FunctionOptions, ConstructorWithVirtualBases),
};

static const EnumEntry<uint8_t> FileChecksumKindNames[] = {
  LLVM_READOBJ_ENUM_CLASS_ENT(FileChecksumKind, None),
  LLVM_READOBJ_ENUM_CLASS_ENT(FileChecksumKind, MD5),
  LLVM_READOBJ_ENUM_CLASS_ENT(FileChecksumKind, SHA1),
  LLVM_READOBJ_ENUM_CLASS_ENT(FileChecksumKind, SHA256),
};

template <typename T>
static std::error_code getSymbolAuxData(const COFFObjectFile *Obj,
                                        COFFSymbolRef Symbol,
                                        uint8_t AuxSymbolIdx, const T *&Aux) {
  ArrayRef<uint8_t> AuxData = Obj->getSymbolAuxData(Symbol);
  AuxData = AuxData.slice(AuxSymbolIdx * Obj->getSymbolTableEntrySize());
  Aux = reinterpret_cast<const T*>(AuxData.data());
  return readobj_error::success;
}

void COFFDumper::cacheRelocations() {
  if (RelocCached)
    return;
  RelocCached = true;

  for (const SectionRef &S : Obj->sections()) {
    const coff_section *Section = Obj->getCOFFSection(S);

    for (const RelocationRef &Reloc : S.relocations())
      RelocMap[Section].push_back(Reloc);

    // Sort relocations by address.
    std::sort(RelocMap[Section].begin(), RelocMap[Section].end(),
              relocAddressLess);
  }
}

void COFFDumper::printDataDirectory(uint32_t Index, const std::string &FieldName) {
  const data_directory *Data;
  if (Obj->getDataDirectory(Index, Data))
    return;
  W.printHex(FieldName + "RVA", Data->RelativeVirtualAddress);
  W.printHex(FieldName + "Size", Data->Size);
}

void COFFDumper::printFileHeaders() {
  time_t TDS = Obj->getTimeDateStamp();
  char FormattedTime[20] = { };
  strftime(FormattedTime, 20, "%Y-%m-%d %H:%M:%S", gmtime(&TDS));

  {
    DictScope D(W, "ImageFileHeader");
    W.printEnum  ("Machine", Obj->getMachine(),
                    makeArrayRef(ImageFileMachineType));
    W.printNumber("SectionCount", Obj->getNumberOfSections());
    W.printHex   ("TimeDateStamp", FormattedTime, Obj->getTimeDateStamp());
    W.printHex   ("PointerToSymbolTable", Obj->getPointerToSymbolTable());
    W.printNumber("SymbolCount", Obj->getNumberOfSymbols());
    W.printNumber("OptionalHeaderSize", Obj->getSizeOfOptionalHeader());
    W.printFlags ("Characteristics", Obj->getCharacteristics(),
                    makeArrayRef(ImageFileCharacteristics));
  }

  // Print PE header. This header does not exist if this is an object file and
  // not an executable.
  const pe32_header *PEHeader = nullptr;
  error(Obj->getPE32Header(PEHeader));
  if (PEHeader)
    printPEHeader<pe32_header>(PEHeader);

  const pe32plus_header *PEPlusHeader = nullptr;
  error(Obj->getPE32PlusHeader(PEPlusHeader));
  if (PEPlusHeader)
    printPEHeader<pe32plus_header>(PEPlusHeader);

  if (const dos_header *DH = Obj->getDOSHeader())
    printDOSHeader(DH);
}

void COFFDumper::printDOSHeader(const dos_header *DH) {
  DictScope D(W, "DOSHeader");
  W.printString("Magic", StringRef(DH->Magic, sizeof(DH->Magic)));
  W.printNumber("UsedBytesInTheLastPage", DH->UsedBytesInTheLastPage);
  W.printNumber("FileSizeInPages", DH->FileSizeInPages);
  W.printNumber("NumberOfRelocationItems", DH->NumberOfRelocationItems);
  W.printNumber("HeaderSizeInParagraphs", DH->HeaderSizeInParagraphs);
  W.printNumber("MinimumExtraParagraphs", DH->MinimumExtraParagraphs);
  W.printNumber("MaximumExtraParagraphs", DH->MaximumExtraParagraphs);
  W.printNumber("InitialRelativeSS", DH->InitialRelativeSS);
  W.printNumber("InitialSP", DH->InitialSP);
  W.printNumber("Checksum", DH->Checksum);
  W.printNumber("InitialIP", DH->InitialIP);
  W.printNumber("InitialRelativeCS", DH->InitialRelativeCS);
  W.printNumber("AddressOfRelocationTable", DH->AddressOfRelocationTable);
  W.printNumber("OverlayNumber", DH->OverlayNumber);
  W.printNumber("OEMid", DH->OEMid);
  W.printNumber("OEMinfo", DH->OEMinfo);
  W.printNumber("AddressOfNewExeHeader", DH->AddressOfNewExeHeader);
}

template <class PEHeader>
void COFFDumper::printPEHeader(const PEHeader *Hdr) {
  DictScope D(W, "ImageOptionalHeader");
  W.printNumber("MajorLinkerVersion", Hdr->MajorLinkerVersion);
  W.printNumber("MinorLinkerVersion", Hdr->MinorLinkerVersion);
  W.printNumber("SizeOfCode", Hdr->SizeOfCode);
  W.printNumber("SizeOfInitializedData", Hdr->SizeOfInitializedData);
  W.printNumber("SizeOfUninitializedData", Hdr->SizeOfUninitializedData);
  W.printHex   ("AddressOfEntryPoint", Hdr->AddressOfEntryPoint);
  W.printHex   ("BaseOfCode", Hdr->BaseOfCode);
  printBaseOfDataField(Hdr);
  W.printHex   ("ImageBase", Hdr->ImageBase);
  W.printNumber("SectionAlignment", Hdr->SectionAlignment);
  W.printNumber("FileAlignment", Hdr->FileAlignment);
  W.printNumber("MajorOperatingSystemVersion",
                Hdr->MajorOperatingSystemVersion);
  W.printNumber("MinorOperatingSystemVersion",
                Hdr->MinorOperatingSystemVersion);
  W.printNumber("MajorImageVersion", Hdr->MajorImageVersion);
  W.printNumber("MinorImageVersion", Hdr->MinorImageVersion);
  W.printNumber("MajorSubsystemVersion", Hdr->MajorSubsystemVersion);
  W.printNumber("MinorSubsystemVersion", Hdr->MinorSubsystemVersion);
  W.printNumber("SizeOfImage", Hdr->SizeOfImage);
  W.printNumber("SizeOfHeaders", Hdr->SizeOfHeaders);
  W.printEnum  ("Subsystem", Hdr->Subsystem, makeArrayRef(PEWindowsSubsystem));
  W.printFlags ("Characteristics", Hdr->DLLCharacteristics,
                makeArrayRef(PEDLLCharacteristics));
  W.printNumber("SizeOfStackReserve", Hdr->SizeOfStackReserve);
  W.printNumber("SizeOfStackCommit", Hdr->SizeOfStackCommit);
  W.printNumber("SizeOfHeapReserve", Hdr->SizeOfHeapReserve);
  W.printNumber("SizeOfHeapCommit", Hdr->SizeOfHeapCommit);
  W.printNumber("NumberOfRvaAndSize", Hdr->NumberOfRvaAndSize);

  if (Hdr->NumberOfRvaAndSize > 0) {
    DictScope D(W, "DataDirectory");
    static const char * const directory[] = {
      "ExportTable", "ImportTable", "ResourceTable", "ExceptionTable",
      "CertificateTable", "BaseRelocationTable", "Debug", "Architecture",
      "GlobalPtr", "TLSTable", "LoadConfigTable", "BoundImport", "IAT",
      "DelayImportDescriptor", "CLRRuntimeHeader", "Reserved"
    };

    for (uint32_t i = 0; i < Hdr->NumberOfRvaAndSize; ++i) {
      printDataDirectory(i, directory[i]);
    }
  }
}

void COFFDumper::printBaseOfDataField(const pe32_header *Hdr) {
  W.printHex("BaseOfData", Hdr->BaseOfData);
}

void COFFDumper::printBaseOfDataField(const pe32plus_header *) {}

void COFFDumper::printCodeViewDebugInfo() {
  // Print types first to build CVUDTNames, then print symbols.
  for (const SectionRef &S : Obj->sections()) {
    StringRef SectionName;
    error(S.getName(SectionName));
    if (SectionName == ".debug$T")
      printCodeViewTypeSection(SectionName, S);
  }
  for (const SectionRef &S : Obj->sections()) {
    StringRef SectionName;
    error(S.getName(SectionName));
    if (SectionName == ".debug$S")
      printCodeViewSymbolSection(SectionName, S);
  }
}

/// Consumes sizeof(T) bytes from the given byte sequence. Returns an error if
/// there are not enough bytes remaining. Reinterprets the consumed bytes as a
/// T object and points 'Res' at them.
template <typename T>
static std::error_code consumeObject(StringRef &Data, const T *&Res) {
  if (Data.size() < sizeof(*Res))
    return object_error::parse_failed;
  Res = reinterpret_cast<const T *>(Data.data());
  Data = Data.drop_front(sizeof(*Res));
  return std::error_code();
}

static std::error_code consumeUInt32(StringRef &Data, uint32_t &Res) {
  const ulittle32_t *IntPtr;
  if (auto EC = consumeObject(Data, IntPtr))
    return EC;
  Res = *IntPtr;
  return std::error_code();
}

void COFFDumper::initializeFileAndStringTables(StringRef Data) {
  while (!Data.empty() && (CVFileChecksumTable.data() == nullptr ||
                           CVStringTable.data() == nullptr)) {
    // The section consists of a number of subsection in the following format:
    // |SubSectionType|SubSectionSize|Contents...|
    uint32_t SubType, SubSectionSize;
    error(consumeUInt32(Data, SubType));
    error(consumeUInt32(Data, SubSectionSize));
    if (SubSectionSize > Data.size())
      return error(object_error::parse_failed);
    switch (ModuleSubstreamKind(SubType)) {
    case ModuleSubstreamKind::FileChecksums:
      CVFileChecksumTable = Data.substr(0, SubSectionSize);
      break;
    case ModuleSubstreamKind::StringTable:
      CVStringTable = Data.substr(0, SubSectionSize);
      break;
    default:
      break;
    }
    Data = Data.drop_front(alignTo(SubSectionSize, 4));
  }
}

void COFFDumper::printCodeViewSymbolSection(StringRef SectionName,
                                            const SectionRef &Section) {
  StringRef SectionContents;
  error(Section.getContents(SectionContents));
  StringRef Data = SectionContents;

  SmallVector<StringRef, 10> FunctionNames;
  StringMap<StringRef> FunctionLineTables;

  ListScope D(W, "CodeViewDebugInfo");
  // Print the section to allow correlation with printSections.
  W.printNumber("Section", SectionName, Obj->getSectionID(Section));

  uint32_t Magic;
  error(consumeUInt32(Data, Magic));
  W.printHex("Magic", Magic);
  if (Magic != COFF::DEBUG_SECTION_MAGIC)
    return error(object_error::parse_failed);

  initializeFileAndStringTables(Data);

  while (!Data.empty()) {
    // The section consists of a number of subsection in the following format:
    // |SubSectionType|SubSectionSize|Contents...|
    uint32_t SubType, SubSectionSize;
    error(consumeUInt32(Data, SubType));
    error(consumeUInt32(Data, SubSectionSize));

    ListScope S(W, "Subsection");
    W.printEnum("SubSectionType", SubType, makeArrayRef(SubSectionTypes));
    W.printHex("SubSectionSize", SubSectionSize);

    // Get the contents of the subsection.
    if (SubSectionSize > Data.size())
      return error(object_error::parse_failed);
    StringRef Contents = Data.substr(0, SubSectionSize);

    // Add SubSectionSize to the current offset and align that offset to find
    // the next subsection.
    size_t SectionOffset = Data.data() - SectionContents.data();
    size_t NextOffset = SectionOffset + SubSectionSize;
    NextOffset = alignTo(NextOffset, 4);
    Data = SectionContents.drop_front(NextOffset);

    // Optionally print the subsection bytes in case our parsing gets confused
    // later.
    if (opts::CodeViewSubsectionBytes)
      printBinaryBlockWithRelocs("SubSectionContents", Section, SectionContents,
                                 Contents);

    switch (ModuleSubstreamKind(SubType)) {
    case ModuleSubstreamKind::Symbols:
      printCodeViewSymbolsSubsection(Contents, Section, SectionContents);
      break;

    case ModuleSubstreamKind::InlineeLines:
      printCodeViewInlineeLines(Contents);
      break;

    case ModuleSubstreamKind::FileChecksums:
      printCodeViewFileChecksums(Contents);
      break;

    case ModuleSubstreamKind::Lines: {
      // Holds a PC to file:line table.  Some data to parse this subsection is
      // stored in the other subsections, so just check sanity and store the
      // pointers for deferred processing.

      if (SubSectionSize < 12) {
        // There should be at least three words to store two function
        // relocations and size of the code.
        error(object_error::parse_failed);
        return;
      }

      StringRef LinkageName;
      error(resolveSymbolName(Obj->getCOFFSection(Section), SectionOffset,
                              LinkageName));
      W.printString("LinkageName", LinkageName);
      if (FunctionLineTables.count(LinkageName) != 0) {
        // Saw debug info for this function already?
        error(object_error::parse_failed);
        return;
      }

      FunctionLineTables[LinkageName] = Contents;
      FunctionNames.push_back(LinkageName);
      break;
    }
    case ModuleSubstreamKind::FrameData: {
      // First four bytes is a relocation against the function.
      const uint32_t *CodePtr;
      error(consumeObject(Contents, CodePtr));
      StringRef LinkageName;
      error(resolveSymbolName(Obj->getCOFFSection(Section), SectionContents,
                              CodePtr, LinkageName));
      W.printString("LinkageName", LinkageName);

      // To find the active frame description, search this array for the
      // smallest PC range that includes the current PC.
      while (!Contents.empty()) {
        const FrameData *FD;
        error(consumeObject(Contents, FD));
        DictScope S(W, "FrameData");
        W.printHex("RvaStart", FD->RvaStart);
        W.printHex("CodeSize", FD->CodeSize);
        W.printHex("LocalSize", FD->LocalSize);
        W.printHex("ParamsSize", FD->ParamsSize);
        W.printHex("MaxStackSize", FD->MaxStackSize);
        W.printString("FrameFunc",
                      CVStringTable.drop_front(FD->FrameFunc).split('\0').first);
        W.printHex("PrologSize", FD->PrologSize);
        W.printHex("SavedRegsSize", FD->SavedRegsSize);
        W.printFlags("Flags", FD->Flags, makeArrayRef(FrameDataFlags));
      }
      break;
    }

    // Do nothing for unrecognized subsections.
    default:
      break;
    }
    W.flush();
  }

  // Dump the line tables now that we've read all the subsections and know all
  // the required information.
  for (unsigned I = 0, E = FunctionNames.size(); I != E; ++I) {
    StringRef Name = FunctionNames[I];
    ListScope S(W, "FunctionLineTable");
    W.printString("LinkageName", Name);

    DataExtractor DE(FunctionLineTables[Name], true, 4);
    uint32_t Offset = 6;  // Skip relocations.
    uint16_t Flags = DE.getU16(&Offset);
    W.printHex("Flags", Flags);
    bool HasColumnInformation = Flags & codeview::LineFlags::HaveColumns;
    uint32_t FunctionSize = DE.getU32(&Offset);
    W.printHex("CodeSize", FunctionSize);
    while (DE.isValidOffset(Offset)) {
      // For each range of lines with the same filename, we have a segment
      // in the line table.  The filename string is accessed using double
      // indirection to the string table subsection using the index subsection.
      uint32_t OffsetInIndex = DE.getU32(&Offset),
               NumLines = DE.getU32(&Offset),
               FullSegmentSize = DE.getU32(&Offset);

      uint32_t ColumnOffset = Offset + 8 * NumLines;
      DataExtractor ColumnDE(DE.getData(), true, 4);

      if (FullSegmentSize !=
          12 + 8 * NumLines + (HasColumnInformation ? 4 * NumLines : 0)) {
        error(object_error::parse_failed);
        return;
      }

      ListScope S(W, "FilenameSegment");
      printFileNameForOffset("Filename", OffsetInIndex);
      for (unsigned LineIdx = 0;
           LineIdx != NumLines && DE.isValidOffset(Offset); ++LineIdx) {
        // Then go the (PC, LineNumber) pairs.  The line number is stored in the
        // least significant 31 bits of the respective word in the table.
        uint32_t PC = DE.getU32(&Offset), LineData = DE.getU32(&Offset);
        if (PC >= FunctionSize) {
          error(object_error::parse_failed);
          return;
        }
        char Buffer[32];
        format("+0x%X", PC).snprint(Buffer, 32);
        ListScope PCScope(W, Buffer);
        LineInfo LI(LineData);
        if (LI.isAlwaysStepInto())
          W.printString("StepInto", StringRef("Always"));
        else if (LI.isNeverStepInto())
          W.printString("StepInto", StringRef("Never"));
        else
          W.printNumber("LineNumberStart", LI.getStartLine());
        W.printNumber("LineNumberEndDelta", LI.getLineDelta());
        W.printBoolean("IsStatement", LI.isStatement());
        if (HasColumnInformation &&
            ColumnDE.isValidOffsetForDataOfSize(ColumnOffset, 4)) {
          uint16_t ColStart = ColumnDE.getU16(&ColumnOffset);
          W.printNumber("ColStart", ColStart);
          uint16_t ColEnd = ColumnDE.getU16(&ColumnOffset);
          W.printNumber("ColEnd", ColEnd);
        }
      }
      // Skip over the column data.
      if (HasColumnInformation) {
        for (unsigned LineIdx = 0;
             LineIdx != NumLines && DE.isValidOffset(Offset); ++LineIdx) {
          DE.getU32(&Offset);
        }
      }
    }
  }
}

static std::error_code decodeNumerictLeaf(StringRef &Data, APSInt &Num) {
  // Used to avoid overload ambiguity on APInt construtor.
  bool FalseVal = false;
  if (Data.size() < 2)
    return object_error::parse_failed;
  uint16_t Short = *reinterpret_cast<const ulittle16_t *>(Data.data());
  Data = Data.drop_front(2);
  if (Short < LF_NUMERIC) {
    Num = APSInt(APInt(/*numBits=*/16, Short, /*isSigned=*/false),
                 /*isUnsigned=*/true);
    return std::error_code();
  }
  switch (Short) {
  case LF_CHAR:
    Num = APSInt(APInt(/*numBits=*/8,
                       *reinterpret_cast<const int8_t *>(Data.data()),
                       /*isSigned=*/true),
                 /*isUnsigned=*/false);
    Data = Data.drop_front(1);
    return std::error_code();
  case LF_SHORT:
    Num = APSInt(APInt(/*numBits=*/16,
                       *reinterpret_cast<const little16_t *>(Data.data()),
                       /*isSigned=*/true),
                 /*isUnsigned=*/false);
    Data = Data.drop_front(2);
    return std::error_code();
  case LF_USHORT:
    Num = APSInt(APInt(/*numBits=*/16,
                       *reinterpret_cast<const ulittle16_t *>(Data.data()),
                       /*isSigned=*/false),
                 /*isUnsigned=*/true);
    Data = Data.drop_front(2);
    return std::error_code();
  case LF_LONG:
    Num = APSInt(APInt(/*numBits=*/32,
                       *reinterpret_cast<const little32_t *>(Data.data()),
                       /*isSigned=*/true),
                 /*isUnsigned=*/false);
    Data = Data.drop_front(4);
    return std::error_code();
  case LF_ULONG:
    Num = APSInt(APInt(/*numBits=*/32,
                       *reinterpret_cast<const ulittle32_t *>(Data.data()),
                       /*isSigned=*/FalseVal),
                 /*isUnsigned=*/true);
    Data = Data.drop_front(4);
    return std::error_code();
  case LF_QUADWORD:
    Num = APSInt(APInt(/*numBits=*/64,
                       *reinterpret_cast<const little64_t *>(Data.data()),
                       /*isSigned=*/true),
                 /*isUnsigned=*/false);
    Data = Data.drop_front(8);
    return std::error_code();
  case LF_UQUADWORD:
    Num = APSInt(APInt(/*numBits=*/64,
                       *reinterpret_cast<const ulittle64_t *>(Data.data()),
                       /*isSigned=*/false),
                 /*isUnsigned=*/true);
    Data = Data.drop_front(8);
    return std::error_code();
  }
  return object_error::parse_failed;
}

/// Decode an unsigned integer numeric leaf value.
std::error_code decodeUIntLeaf(StringRef &Data, uint64_t &Num) {
  APSInt N;
  if (std::error_code err = decodeNumerictLeaf(Data, N))
    return err;
  if (N.isSigned() || !N.isIntN(64))
    return object_error::parse_failed;
  Num = N.getLimitedValue();
  return std::error_code();
}

void COFFDumper::printCodeViewSymbolsSubsection(StringRef Subsection,
                                                const SectionRef &Section,
                                                StringRef SectionContents) {
  if (Subsection.size() < sizeof(SymRecord))
    return error(object_error::parse_failed);

  const coff_section *Sec = Obj->getCOFFSection(Section);

  // This holds the remaining data to parse.
  StringRef Data = Subsection;

  bool InFunctionScope = false;
  while (!Data.empty()) {
    const SymRecord *Rec;
    error(consumeObject(Data, Rec));

    StringRef SymData = Data.substr(0, Rec->RecordLength - 2);
    StringRef OrigSymData = SymData;

    Data = Data.drop_front(Rec->RecordLength - 2);

    SymbolRecordKind Kind = Rec->getKind();
    switch (Kind) {
    case S_LPROC32:
    case S_GPROC32:
    case S_GPROC32_ID:
    case S_LPROC32_ID:
    case S_LPROC32_DPC:
    case S_LPROC32_DPC_ID: {
      DictScope S(W, "ProcStart");
      const ProcSym *Proc;
      error(consumeObject(SymData, Proc));
      if (InFunctionScope)
        return error(object_error::parse_failed);
      InFunctionScope = true;

      StringRef LinkageName;
      StringRef DisplayName = SymData.split('\0').first;
      W.printHex("PtrParent", Proc->PtrParent);
      W.printHex("PtrEnd", Proc->PtrEnd);
      W.printHex("PtrNext", Proc->PtrNext);
      W.printHex("CodeSize", Proc->CodeSize);
      W.printHex("DbgStart", Proc->DbgStart);
      W.printHex("DbgEnd", Proc->DbgEnd);
      printTypeIndex("FunctionType", Proc->FunctionType);
      printRelocatedField("CodeOffset", Sec, SectionContents, &Proc->CodeOffset,
                          &LinkageName);
      W.printHex("Segment", Proc->Segment);
      W.printFlags("Flags", Proc->Flags, makeArrayRef(ProcSymFlags));
      W.printString("DisplayName", DisplayName);
      W.printString("LinkageName", LinkageName);
      break;
    }

    case S_PROC_ID_END: {
      W.startLine() << "ProcEnd\n";
      InFunctionScope = false;
      break;
    }

    case S_BLOCK32: {
      DictScope S(W, "BlockStart");
      const BlockSym *Block;
      error(consumeObject(SymData, Block));

      StringRef BlockName = SymData.split('\0').first;
      StringRef LinkageName;
      W.printHex("PtrParent", Block->PtrParent);
      W.printHex("PtrEnd", Block->PtrEnd);
      W.printHex("CodeSize", Block->CodeSize);
      printRelocatedField("CodeOffset", Sec, SectionContents,
                          &Block->CodeOffset, &LinkageName);
      W.printHex("Segment", Block->Segment);
      W.printString("BlockName", BlockName);
      W.printString("LinkageName", LinkageName);
      break;
    }

    case S_END: {
      W.startLine() << "BlockEnd\n";
      InFunctionScope = false;
      break;
    }

    case S_LABEL32: {
      DictScope S(W, "Label");
      const LabelSym *Label;
      error(consumeObject(SymData, Label));

      StringRef DisplayName = SymData.split('\0').first;
      StringRef LinkageName;
      printRelocatedField("CodeOffset", Sec, SectionContents,
                          &Label->CodeOffset, &LinkageName);
      W.printHex("Segment", Label->Segment);
      W.printHex("Flags", Label->Flags);
      W.printFlags("Flags", Label->Flags, makeArrayRef(ProcSymFlags));
      W.printString("DisplayName", DisplayName);
      W.printString("LinkageName", LinkageName);
      break;
    }

    case S_INLINESITE: {
      DictScope S(W, "InlineSite");
      const InlineSiteSym *InlineSite;
      error(consumeObject(SymData, InlineSite));
      W.printHex("PtrParent", InlineSite->PtrParent);
      W.printHex("PtrEnd", InlineSite->PtrEnd);
      printTypeIndex("Inlinee", InlineSite->Inlinee);

      auto GetCompressedAnnotation = [&]() -> uint32_t {
        if (SymData.empty())
          return -1;

        uint8_t FirstByte = SymData.front();
        SymData = SymData.drop_front();

        if ((FirstByte & 0x80) == 0x00)
          return FirstByte;

        if (SymData.empty())
          return -1;

        uint8_t SecondByte = SymData.front();
        SymData = SymData.drop_front();

        if ((FirstByte & 0xC0) == 0x80)
          return ((FirstByte & 0x3F) << 8) | SecondByte;

        if (SymData.empty())
          return -1;

        uint8_t ThirdByte = SymData.front();
        SymData = SymData.drop_front();

        if (SymData.empty())
          return -1;

        uint8_t FourthByte = SymData.front();
        SymData = SymData.drop_front();

        if ((FirstByte & 0xE0) == 0xC0)
          return ((FirstByte & 0x1F) << 24) | (SecondByte << 16) |
                 (ThirdByte << 8) | FourthByte;

        return -1;
      };
      auto DecodeSignedOperand = [](uint32_t Operand) -> int32_t {
        if (Operand & 1)
          return -(Operand >> 1);
        return Operand >> 1;
      };

      ListScope BinaryAnnotations(W, "BinaryAnnotations");
      while (!SymData.empty()) {
        uint32_t OpCode = GetCompressedAnnotation();
        switch (OpCode) {
        default:
        case Invalid:
          return error(object_error::parse_failed);
        case CodeOffset:
          W.printHex("CodeOffset", GetCompressedAnnotation());
          break;
        case ChangeCodeOffsetBase:
          W.printNumber("ChangeCodeOffsetBase", GetCompressedAnnotation());
          break;
        case ChangeCodeOffset:
          W.printHex("ChangeCodeOffset", GetCompressedAnnotation());
          break;
        case ChangeCodeLength:
          W.printHex("ChangeCodeLength", GetCompressedAnnotation());
          break;
        case ChangeFile:
          printFileNameForOffset("ChangeFile", GetCompressedAnnotation());
          break;
        case ChangeLineOffset:
          W.printNumber("ChangeLineOffset",
                        DecodeSignedOperand(GetCompressedAnnotation()));
          break;
        case ChangeLineEndDelta:
          W.printNumber("ChangeLineEndDelta", GetCompressedAnnotation());
          break;
        case ChangeRangeKind:
          W.printNumber("ChangeRangeKind", GetCompressedAnnotation());
          break;
        case ChangeColumnStart:
          W.printNumber("ChangeColumnStart", GetCompressedAnnotation());
          break;
        case ChangeColumnEndDelta:
          W.printNumber("ChangeColumnEndDelta",
                        DecodeSignedOperand(GetCompressedAnnotation()));
          break;
        case ChangeCodeOffsetAndLineOffset: {
          uint32_t Annotation = GetCompressedAnnotation();
          int32_t LineOffset = DecodeSignedOperand(Annotation >> 4);
          uint32_t CodeOffset = Annotation & 0xf;
          W.startLine() << "ChangeCodeOffsetAndLineOffset: {CodeOffset: "
                        << W.hex(CodeOffset) << ", LineOffset: " << LineOffset
                        << "}\n";
          break;
        }
        case ChangeCodeLengthAndCodeOffset: {
          uint32_t Length = GetCompressedAnnotation();
          uint32_t CodeOffset = GetCompressedAnnotation();
          W.startLine() << "ChangeCodeLengthAndCodeOffset: {CodeOffset: "
                        << W.hex(CodeOffset) << ", Length: " << W.hex(Length)
                        << "}\n";
          break;
        }
        case ChangeColumnEnd:
          W.printNumber("ChangeColumnEnd", GetCompressedAnnotation());
          break;
        }
      }
      break;
    }

    case S_INLINESITE_END: {
      DictScope S(W, "InlineSiteEnd");
      break;
    }

    case S_CALLERS:
    case S_CALLEES: {
      ListScope S(W, Kind == S_CALLEES ? "Callees" : "Callers");
      uint32_t Count;
      error(consumeUInt32(SymData, Count));
      for (uint32_t I = 0; I < Count; ++I) {
        const TypeIndex *FuncID;
        error(consumeObject(SymData, FuncID));
        printTypeIndex("FuncID", *FuncID);
      }
      break;
    }

    case S_LOCAL: {
      DictScope S(W, "Local");
      const LocalSym *Local;
      error(consumeObject(SymData, Local));
      printTypeIndex("Type", Local->Type);
      W.printFlags("Flags", uint16_t(Local->Flags), makeArrayRef(LocalFlags));
      StringRef VarName = SymData.split('\0').first;
      W.printString("VarName", VarName);
      break;
    }

    case S_DEFRANGE: {
      DictScope S(W, "DefRange");
      const DefRangeSym *DefRange;
      error(consumeObject(SymData, DefRange));
      W.printString(
          "Program",
          CVStringTable.drop_front(DefRange->Program).split('\0').first);
      printLocalVariableAddrRange(DefRange->Range, Sec, SectionContents);
      printLocalVariableAddrGap(SymData);
      break;
    }
    case S_DEFRANGE_SUBFIELD: {
      DictScope S(W, "DefRangeSubfield");
      const DefRangeSubfieldSym *DefRangeSubfield;
      error(consumeObject(SymData, DefRangeSubfield));
      W.printString("Program",
                    CVStringTable.drop_front(DefRangeSubfield->Program)
                        .split('\0')
                        .first);
      W.printNumber("OffsetInParent", DefRangeSubfield->OffsetInParent);
      printLocalVariableAddrRange(DefRangeSubfield->Range, Sec,
                                  SectionContents);
      printLocalVariableAddrGap(SymData);
      break;
    }
    case S_DEFRANGE_REGISTER: {
      DictScope S(W, "DefRangeRegister");
      const DefRangeRegisterSym *DefRangeRegister;
      error(consumeObject(SymData, DefRangeRegister));
      W.printNumber("Register", DefRangeRegister->Register);
      W.printNumber("MayHaveNoName", DefRangeRegister->MayHaveNoName);
      printLocalVariableAddrRange(DefRangeRegister->Range, Sec,
                                  SectionContents);
      printLocalVariableAddrGap(SymData);
      break;
    }
    case S_DEFRANGE_SUBFIELD_REGISTER: {
      DictScope S(W, "DefRangeSubfieldRegister");
      const DefRangeSubfieldRegisterSym *DefRangeSubfieldRegisterSym;
      error(consumeObject(SymData, DefRangeSubfieldRegisterSym));
      W.printNumber("Register", DefRangeSubfieldRegisterSym->Register);
      W.printNumber("MayHaveNoName",
                    DefRangeSubfieldRegisterSym->MayHaveNoName);
      W.printNumber("OffsetInParent",
                    DefRangeSubfieldRegisterSym->OffsetInParent);
      printLocalVariableAddrRange(DefRangeSubfieldRegisterSym->Range, Sec,
                                  SectionContents);
      printLocalVariableAddrGap(SymData);
      break;
    }
    case S_DEFRANGE_FRAMEPOINTER_REL: {
      DictScope S(W, "DefRangeFramePointerRel");
      const DefRangeFramePointerRelSym *DefRangeFramePointerRel;
      error(consumeObject(SymData, DefRangeFramePointerRel));
      W.printNumber("Offset", DefRangeFramePointerRel->Offset);
      printLocalVariableAddrRange(DefRangeFramePointerRel->Range, Sec,
                                  SectionContents);
      printLocalVariableAddrGap(SymData);
      break;
    }
    case S_DEFRANGE_FRAMEPOINTER_REL_FULL_SCOPE: {
      DictScope S(W, "DefRangeFramePointerRelFullScope");
      const DefRangeFramePointerRelFullScopeSym
          *DefRangeFramePointerRelFullScope;
      error(consumeObject(SymData, DefRangeFramePointerRelFullScope));
      W.printNumber("Offset", DefRangeFramePointerRelFullScope->Offset);
      break;
    }
    case S_DEFRANGE_REGISTER_REL: {
      DictScope S(W, "DefRangeRegisterRel");
      const DefRangeRegisterRelSym *DefRangeRegisterRel;
      error(consumeObject(SymData, DefRangeRegisterRel));
      W.printNumber("BaseRegister", DefRangeRegisterRel->BaseRegister);
      W.printBoolean("HasSpilledUDTMember",
                     DefRangeRegisterRel->hasSpilledUDTMember());
      W.printNumber("OffsetInParent", DefRangeRegisterRel->offsetInParent());
      printLocalVariableAddrRange(DefRangeRegisterRel->Range, Sec,
                                  SectionContents);
      printLocalVariableAddrGap(SymData);
      break;
    }

    case S_CALLSITEINFO: {
      DictScope S(W, "CallSiteInfo");
      const CallSiteInfoSym *CallSiteInfo;
      error(consumeObject(SymData, CallSiteInfo));

      StringRef LinkageName;
      printRelocatedField("CodeOffset", Sec, SectionContents,
                          &CallSiteInfo->CodeOffset, &LinkageName);
      W.printHex("Segment", CallSiteInfo->Segment);
      W.printHex("Reserved", CallSiteInfo->Reserved);
      printTypeIndex("Type", CallSiteInfo->Type);
      W.printString("LinkageName", LinkageName);
      break;
    }

    case S_HEAPALLOCSITE: {
      DictScope S(W, "HeapAllocationSite");
      const HeapAllocationSiteSym *HeapAllocationSite;
      error(consumeObject(SymData, HeapAllocationSite));

      StringRef LinkageName;
      printRelocatedField("CodeOffset", Sec, SectionContents,
                          &HeapAllocationSite->CodeOffset, &LinkageName);
      W.printHex("Segment", HeapAllocationSite->Segment);
      W.printHex("CallInstructionSize",
                 HeapAllocationSite->CallInstructionSize);
      printTypeIndex("Type", HeapAllocationSite->Type);
      W.printString("LinkageName", LinkageName);
      break;
    }

    case S_FRAMECOOKIE: {
      DictScope S(W, "FrameCookie");
      const FrameCookieSym *FrameCookie;
      error(consumeObject(SymData, FrameCookie));

      StringRef LinkageName;
      printRelocatedField("CodeOffset", Sec, SectionContents,
                          &FrameCookie->CodeOffset, &LinkageName);
      W.printHex("Register", FrameCookie->Register);
      W.printEnum("CookieKind", uint16_t(FrameCookie->CookieKind),
                  makeArrayRef(FrameCookieKinds));
      break;
    }

    case S_LDATA32:
    case S_GDATA32:
    case S_LMANDATA:
    case S_GMANDATA: {
      DictScope S(W, "DataSym");
      const DataSym *Data;
      error(consumeObject(SymData, Data));

      StringRef DisplayName = SymData.split('\0').first;
      StringRef LinkageName;
      printRelocatedField("DataOffset", Sec, SectionContents, &Data->DataOffset,
                          &LinkageName);
      printTypeIndex("Type", Data->Type);
      W.printString("DisplayName", DisplayName);
      W.printString("LinkageName", LinkageName);
      break;
    }

    case S_LTHREAD32:
    case S_GTHREAD32: {
      DictScope S(W, "ThreadLocalDataSym");
      const ThreadLocalDataSym *Data;
      error(consumeObject(SymData, Data));

      StringRef DisplayName = SymData.split('\0').first;
      StringRef LinkageName;
      printRelocatedField("DataOffset", Sec, SectionContents, &Data->DataOffset,
                          &LinkageName);
      printTypeIndex("Type", Data->Type);
      W.printString("DisplayName", DisplayName);
      W.printString("LinkageName", LinkageName);
      break;
    }

    case S_OBJNAME: {
      DictScope S(W, "ObjectName");
      const ObjNameSym *ObjName;
      error(consumeObject(SymData, ObjName));
      W.printHex("Signature", ObjName->Signature);
      StringRef ObjectName = SymData.split('\0').first;
      W.printString("ObjectName", ObjectName);
      break;
    }

    case S_COMPILE3: {
      DictScope S(W, "CompilerFlags");
      const CompileSym3 *CompFlags;
      error(consumeObject(SymData, CompFlags));
      W.printEnum("Language", CompFlags->getLanguage(),
                  makeArrayRef(SourceLanguages));
      W.printFlags("Flags", CompFlags->flags & ~0xff,
                   makeArrayRef(CompileSym3Flags));
      W.printEnum("Machine", unsigned(CompFlags->Machine),
                  makeArrayRef(CPUTypeNames));
      std::string FrontendVersion;
      {
        raw_string_ostream Out(FrontendVersion);
        Out << CompFlags->VersionFrontendMajor << '.'
            << CompFlags->VersionFrontendMinor << '.'
            << CompFlags->VersionFrontendBuild << '.'
            << CompFlags->VersionFrontendQFE;
      }
      std::string BackendVersion;
      {
        raw_string_ostream Out(BackendVersion);
        Out << CompFlags->VersionBackendMajor << '.'
            << CompFlags->VersionBackendMinor << '.'
            << CompFlags->VersionBackendBuild << '.'
            << CompFlags->VersionBackendQFE;
      }
      W.printString("FrontendVersion", FrontendVersion);
      W.printString("BackendVersion", BackendVersion);
      StringRef VersionName = SymData.split('\0').first;
      W.printString("VersionName", VersionName);
      break;
    }

    case S_FRAMEPROC: {
      DictScope S(W, "FrameProc");
      const FrameProcSym *FrameProc;
      error(consumeObject(SymData, FrameProc));
      W.printHex("TotalFrameBytes", FrameProc->TotalFrameBytes);
      W.printHex("PaddingFrameBytes", FrameProc->PaddingFrameBytes);
      W.printHex("OffsetToPadding", FrameProc->OffsetToPadding);
      W.printHex("BytesOfCalleeSavedRegisters", FrameProc->BytesOfCalleeSavedRegisters);
      W.printHex("OffsetOfExceptionHandler", FrameProc->OffsetOfExceptionHandler);
      W.printHex("SectionIdOfExceptionHandler", FrameProc->SectionIdOfExceptionHandler);
      W.printFlags("Flags", FrameProc->Flags, makeArrayRef(FrameProcSymFlags));
      break;
    }

    case S_UDT:
    case S_COBOLUDT: {
      DictScope S(W, "UDT");
      const UDTSym *UDT;
      error(consumeObject(SymData, UDT));
      printTypeIndex("Type", UDT->Type);
      StringRef UDTName = SymData.split('\0').first;
      W.printString("UDTName", UDTName);
      break;
    }

    case S_BPREL32: {
      DictScope S(W, "BPRelativeSym");
      const BPRelativeSym *BPRel;
      error(consumeObject(SymData, BPRel));
      W.printNumber("Offset", BPRel->Offset);
      printTypeIndex("Type", BPRel->Type);
      StringRef VarName = SymData.split('\0').first;
      W.printString("VarName", VarName);
      break;
    }

    case S_REGREL32: {
      DictScope S(W, "RegRelativeSym");
      const RegRelativeSym *RegRel;
      error(consumeObject(SymData, RegRel));
      W.printHex("Offset", RegRel->Offset);
      printTypeIndex("Type", RegRel->Type);
      W.printHex("Register", RegRel->Register);
      StringRef VarName = SymData.split('\0').first;
      W.printString("VarName", VarName);
      break;
    }

    case S_BUILDINFO: {
      DictScope S(W, "BuildInfo");
      const BuildInfoSym *BuildInfo;
      error(consumeObject(SymData, BuildInfo));
      W.printNumber("BuildId", BuildInfo->BuildId);
      break;
    }

    case S_CONSTANT:
    case S_MANCONSTANT: {
      DictScope S(W, "Constant");
      const ConstantSym *Constant;
      error(consumeObject(SymData, Constant));
      printTypeIndex("Type", Constant->Type);
      APSInt Value;
      error(decodeNumerictLeaf(SymData, Value));
      W.printNumber("Value", Value);
      StringRef Name = SymData.split('\0').first;
      W.printString("Name", Name);
      break;
    }

    default: {
      DictScope S(W, "UnknownSym");
      W.printHex("Kind", unsigned(Kind));
      W.printHex("Size", Rec->RecordLength);
      break;
    }
    }

    if (opts::CodeViewSubsectionBytes)
      printBinaryBlockWithRelocs("SymData", Section, SectionContents,
                                 OrigSymData);
    W.flush();
  }
  W.flush();
}

void COFFDumper::printCodeViewFileChecksums(StringRef Subsection) {
  StringRef Data = Subsection;
  while (!Data.empty()) {
    DictScope S(W, "FileChecksum");
    const FileChecksum *FC;
    error(consumeObject(Data, FC));
    if (FC->FileNameOffset >= CVStringTable.size())
      error(object_error::parse_failed);
    StringRef Filename =
        CVStringTable.drop_front(FC->FileNameOffset).split('\0').first;
    W.printHex("Filename", Filename, FC->FileNameOffset);
    W.printHex("ChecksumSize", FC->ChecksumSize);
    W.printEnum("ChecksumKind", uint8_t(FC->ChecksumKind),
                makeArrayRef(FileChecksumKindNames));
    if (FC->ChecksumSize >= Data.size())
      error(object_error::parse_failed);
    StringRef ChecksumBytes = Data.substr(0, FC->ChecksumSize);
    W.printBinary("ChecksumBytes", ChecksumBytes);
    unsigned PaddedSize = alignTo(FC->ChecksumSize + sizeof(FileChecksum), 4) -
                          sizeof(FileChecksum);
    Data = Data.drop_front(PaddedSize);
  }
}

void COFFDumper::printCodeViewInlineeLines(StringRef Subsection) {
  StringRef Data = Subsection;
  uint32_t Signature;
  error(consumeUInt32(Data, Signature));
  bool HasExtraFiles = Signature == unsigned(InlineeLinesSignature::ExtraFiles);

  while (!Data.empty()) {
    const InlineeSourceLine *ISL;
    error(consumeObject(Data, ISL));
    DictScope S(W, "InlineeSourceLine");
    printTypeIndex("Inlinee", ISL->Inlinee);
    printFileNameForOffset("FileID", ISL->FileID);
    W.printNumber("SourceLineNum", ISL->SourceLineNum);

    if (HasExtraFiles) {
      uint32_t ExtraFileCount;
      error(consumeUInt32(Data, ExtraFileCount));
      W.printNumber("ExtraFileCount", ExtraFileCount);
      ListScope ExtraFiles(W, "ExtraFiles");
      for (unsigned I = 0; I < ExtraFileCount; ++I) {
        uint32_t FileID;
        error(consumeUInt32(Data, FileID));
        printFileNameForOffset("FileID", FileID);
      }
    }
  }
}

StringRef getRemainingTypeBytes(const TypeRecordPrefix *Rec, const char *Start) {
  ptrdiff_t StartOffset = Start - reinterpret_cast<const char *>(Rec);
  size_t RecSize = Rec->Len + 2;
  assert(StartOffset >= 0 && "negative start-offset!");
  assert(static_cast<size_t>(StartOffset) <= RecSize &&
         "Start beyond the end of Rec");
  return StringRef(Start, RecSize - StartOffset);
}

StringRef getRemainingBytesAsString(const TypeRecordPrefix *Rec, const char *Start) {
  StringRef Remaining = getRemainingTypeBytes(Rec, Start);
  StringRef Leading, Trailing;
  std::tie(Leading, Trailing) = Remaining.split('\0');
  return Leading;
}

StringRef COFFDumper::getTypeName(TypeIndex TI) {
  if (TI.isNoType())
    return "<no type>";

  if (TI.isSimple()) {
    // This is a simple type.
    for (const auto &SimpleTypeName : SimpleTypeNames) {
      if (SimpleTypeName.Value == TI.getSimpleKind()) {
        if (TI.getSimpleMode() == SimpleTypeMode::Direct)
          return SimpleTypeName.Name.drop_back(1);
        // Otherwise, this is a pointer type. We gloss over the distinction
        // between near, far, 64, 32, etc, and just give a pointer type.
        return SimpleTypeName.Name;
      }
    }
    return "<unknown simple type>";
  }

  // User-defined type.
  StringRef UDTName;
  unsigned UDTIndex = TI.getIndex() - 0x1000;
  if (UDTIndex < CVUDTNames.size())
    return CVUDTNames[UDTIndex];

  return "<unknown UDT>";
}

void COFFDumper::printTypeIndex(StringRef FieldName, TypeIndex TI) {
  StringRef TypeName;
  if (!TI.isNoType())
    TypeName = getTypeName(TI);
  if (!TypeName.empty())
    W.printHex(FieldName, TypeName, TI.getIndex());
  else
    W.printHex(FieldName, TI.getIndex());
}

void COFFDumper::printLocalVariableAddrRange(
    const LocalVariableAddrRange &Range, const coff_section *Sec,
    StringRef SectionContents) {
  DictScope S(W, "LocalVariableAddrRange");
  printRelocatedField("OffsetStart", Sec, SectionContents, &Range.OffsetStart);
  W.printHex("ISectStart", Range.ISectStart);
  W.printNumber("Range", Range.Range);
}

void COFFDumper::printLocalVariableAddrGap(StringRef &SymData) {
  while (!SymData.empty()) {
    const LocalVariableAddrGap *Gap;
    error(consumeObject(SymData, Gap));
    ListScope S(W, "LocalVariableAddrGap");
    W.printNumber("GapStartOffset", Gap->GapStartOffset);
    W.printNumber("Range", Gap->Range);
  }
}

StringRef COFFDumper::getFileNameForFileOffset(uint32_t FileOffset) {
  // The file checksum subsection should precede all references to it.
  if (!CVFileChecksumTable.data() || !CVStringTable.data())
    error(object_error::parse_failed);
  // Check if the file checksum table offset is valid.
  if (FileOffset >= CVFileChecksumTable.size())
    error(object_error::parse_failed);

  // The string table offset comes first before the file checksum.
  StringRef Data = CVFileChecksumTable.drop_front(FileOffset);
  uint32_t StringOffset;
  error(consumeUInt32(Data, StringOffset));

  // Check if the string table offset is valid.
  if (StringOffset >= CVStringTable.size())
    error(object_error::parse_failed);

  // Return the null-terminated string.
  return CVStringTable.drop_front(StringOffset).split('\0').first;
}

void COFFDumper::printFileNameForOffset(StringRef Label, uint32_t FileOffset) {
  W.printHex(Label, getFileNameForFileOffset(FileOffset), FileOffset);
}

static StringRef getLeafTypeName(TypeLeafKind LT) {
  switch (LT) {
  case LF_STRING_ID: return "StringId";
  case LF_FIELDLIST: return "FieldList";
  case LF_ARGLIST:
  case LF_SUBSTR_LIST: return "ArgList";
  case LF_CLASS:
  case LF_STRUCTURE:
  case LF_INTERFACE: return "ClassType";
  case LF_UNION: return "UnionType";
  case LF_ENUM: return "EnumType";
  case LF_ARRAY: return "ArrayType";
  case LF_VFTABLE: return "VFTableType";
  case LF_MFUNC_ID: return "MemberFuncId";
  case LF_PROCEDURE: return "ProcedureType";
  case LF_MFUNCTION: return "MemberFunctionType";
  case LF_METHODLIST: return "MethodListEntry";
  case LF_FUNC_ID: return "FuncId";
  case LF_TYPESERVER2: return "TypeServer2";
  case LF_POINTER: return "PointerType";
  case LF_MODIFIER: return "TypeModifier";
  case LF_VTSHAPE: return "VTableShape";
  case LF_UDT_SRC_LINE: return "UDTSrcLine";
  case LF_BUILDINFO: return "BuildInfo";
  default: break;
  }
  return "UnknownLeaf";
}

void COFFDumper::printCodeViewTypeSection(StringRef SectionName,
                                          const SectionRef &Section) {
  ListScope D(W, "CodeViewTypes");
  W.printNumber("Section", SectionName, Obj->getSectionID(Section));
  StringRef Data;
  error(Section.getContents(Data));
  W.printBinaryBlock("Data", Data);

  unsigned Magic = *reinterpret_cast<const ulittle32_t *>(Data.data());
  W.printHex("Magic", Magic);

  Data = Data.drop_front(4);

  while (!Data.empty()) {
    const TypeRecordPrefix *Rec;
    error(consumeObject(Data, Rec));
    auto Leaf = static_cast<TypeLeafKind>(uint16_t(Rec->Leaf));

    // This record is 'Len - 2' bytes, and the next one starts immediately
    // afterwards.
    StringRef LeafData = Data.substr(0, Rec->Len - 2);
    StringRef RemainingData = Data.drop_front(LeafData.size());

    // Find the name of this leaf type.
    StringRef LeafName = getLeafTypeName(Leaf);
    DictScope S(W, LeafName);
    unsigned NextTypeIndex = 0x1000 + CVUDTNames.size();
    W.printEnum("TypeLeafKind", unsigned(Leaf), makeArrayRef(LeafTypeNames));
    W.printHex("TypeIndex", NextTypeIndex);

    // Fill this in inside the switch to get something in CVUDTNames.
    StringRef Name;

    switch (Leaf) {
    default: {
      W.printHex("Size", Rec->Len);
      break;
    }

    case LF_STRING_ID: {
      const StringId *String;
      error(consumeObject(LeafData, String));
      W.printHex("Id", String->id.getIndex());
      StringRef StringData = getRemainingBytesAsString(Rec, LeafData.data());
      W.printString("StringData", StringData);
      // Put this in CVUDTNames so it gets printed with LF_UDT_SRC_LINE.
      Name = StringData;
      break;
    }

    case LF_FIELDLIST: {
      W.printHex("Size", Rec->Len);
      // FieldList has no fixed prefix that can be described with a struct. All
      // the bytes must be interpreted as more records.
      printCodeViewFieldList(LeafData);
      break;
    }

    case LF_ARGLIST:
    case LF_SUBSTR_LIST: {
      const ArgList *Args;
      error(consumeObject(LeafData, Args));
      W.printNumber("NumArgs", Args->NumArgs);
      ListScope Arguments(W, "Arguments");
      SmallString<256> TypeName("(");
      for (uint32_t ArgI = 0; ArgI != Args->NumArgs; ++ArgI) {
        const TypeIndex *Type;
        error(consumeObject(LeafData, Type));
        printTypeIndex("ArgType", *Type);
        StringRef ArgTypeName = getTypeName(*Type);
        TypeName.append(ArgTypeName);
        if (ArgI + 1 != Args->NumArgs)
          TypeName.append(", ");
      }
      TypeName.push_back(')');
      Name = TypeNames.insert(TypeName).first->getKey();
      break;
    }

    case LF_CLASS:
    case LF_STRUCTURE:
    case LF_INTERFACE: {
      const ClassType *Class;
      error(consumeObject(LeafData, Class));
      W.printNumber("MemberCount", Class->MemberCount);
      uint16_t Props = Class->Properties;
      W.printFlags("Properties", Props, makeArrayRef(ClassOptionNames));
      printTypeIndex("FieldList", Class->FieldList);
      printTypeIndex("DerivedFrom", Class->DerivedFrom);
      printTypeIndex("VShape", Class->VShape);
      uint64_t SizeOf;
      error(decodeUIntLeaf(LeafData, SizeOf));
      W.printNumber("SizeOf", SizeOf);
      StringRef LinkageName;
      std::tie(Name, LinkageName) = LeafData.split('\0');
      W.printString("Name", Name);
      if (Props & uint16_t(ClassOptions::HasUniqueName)) {
        LinkageName = getRemainingBytesAsString(Rec, LinkageName.data());
        if (LinkageName.empty())
          return error(object_error::parse_failed);
        W.printString("LinkageName", LinkageName);
      }
      break;
    }

    case LF_UNION: {
      const UnionType *Union;
      error(consumeObject(LeafData, Union));
      W.printNumber("MemberCount", Union->MemberCount);
      uint16_t Props = Union->Properties;
      W.printFlags("Properties", Props, makeArrayRef(ClassOptionNames));
      printTypeIndex("FieldList", Union->FieldList);
      uint64_t SizeOf;
      error(decodeUIntLeaf(LeafData, SizeOf));
      W.printNumber("SizeOf", SizeOf);
      StringRef LinkageName;
      std::tie(Name, LinkageName) = LeafData.split('\0');
      W.printString("Name", Name);
      if (Props & uint16_t(ClassOptions::HasUniqueName)) {
        LinkageName = getRemainingBytesAsString(Rec, LinkageName.data());
        if (LinkageName.empty())
          return error(object_error::parse_failed);
        W.printString("LinkageName", LinkageName);
      }
      break;
    }

    case LF_ENUM: {
      const EnumType *Enum;
      error(consumeObject(LeafData, Enum));
      W.printNumber("NumEnumerators", Enum->NumEnumerators);
      W.printFlags("Properties", uint16_t(Enum->Properties),
                   makeArrayRef(ClassOptionNames));
      printTypeIndex("UnderlyingType", Enum->UnderlyingType);
      printTypeIndex("FieldListType", Enum->FieldListType);
      Name = LeafData.split('\0').first;
      W.printString("Name", Name);
      break;
    }

    case LF_ARRAY: {
      const ArrayType *AT;
      error(consumeObject(LeafData, AT));
      printTypeIndex("ElementType", AT->ElementType);
      printTypeIndex("IndexType", AT->IndexType);
      uint64_t SizeOf;
      error(decodeUIntLeaf(LeafData, SizeOf));
      W.printNumber("SizeOf", SizeOf);
      Name = LeafData.split('\0').first;
      W.printString("Name", Name);
      break;
    }

    case LF_VFTABLE: {
      const VFTableType *VFT;
      error(consumeObject(LeafData, VFT));
      printTypeIndex("CompleteClass", VFT->CompleteClass);
      printTypeIndex("OverriddenVFTable", VFT->OverriddenVFTable);
      W.printHex("VFPtrOffset", VFT->VFPtrOffset);
      StringRef NamesData = LeafData.substr(0, VFT->NamesLen);
      std::tie(Name, NamesData) = NamesData.split('\0');
      W.printString("VFTableName", Name);
      while (!NamesData.empty()) {
        StringRef MethodName;
        std::tie(MethodName, NamesData) = NamesData.split('\0');
        W.printString("MethodName", MethodName);
      }
      break;
    }

    case LF_MFUNC_ID: {
      const MemberFuncId *Id;
      error(consumeObject(LeafData, Id));
      printTypeIndex("ClassType", Id->ClassType);
      printTypeIndex("FunctionType", Id->FunctionType);
      Name = LeafData.split('\0').first;
      W.printString("Name", Name);
      break;
    }

    case LF_PROCEDURE: {
      const ProcedureType *Proc;
      error(consumeObject(LeafData, Proc));
      printTypeIndex("ReturnType", Proc->ReturnType);
      W.printEnum("CallingConvention", uint8_t(Proc->CallConv),
                  makeArrayRef(CallingConventions));
      W.printFlags("FunctionOptions", uint8_t(Proc->Options),
                   makeArrayRef(FunctionOptionEnum));
      W.printNumber("NumParameters", Proc->NumParameters);
      printTypeIndex("ArgListType", Proc->ArgListType);

      StringRef ReturnTypeName = getTypeName(Proc->ReturnType);
      StringRef ArgListTypeName = getTypeName(Proc->ArgListType);
      SmallString<256> TypeName(ReturnTypeName);
      TypeName.push_back(' ');
      TypeName.append(ArgListTypeName);
      Name = TypeNames.insert(TypeName).first->getKey();
      break;
    }

    case LF_MFUNCTION: {
      const MemberFunctionType *MemberFunc;
      error(consumeObject(LeafData, MemberFunc));
      printTypeIndex("ReturnType", MemberFunc->ReturnType);
      printTypeIndex("ClassType", MemberFunc->ClassType);
      printTypeIndex("ThisType", MemberFunc->ThisType);
      W.printEnum("CallingConvention", uint8_t(MemberFunc->CallConv),
                  makeArrayRef(CallingConventions));
      W.printFlags("FunctionOptions", uint8_t(MemberFunc->Options),
                   makeArrayRef(FunctionOptionEnum));
      W.printNumber("NumParameters", MemberFunc->NumParameters);
      printTypeIndex("ArgListType", MemberFunc->ArgListType);
      W.printNumber("ThisAdjustment", MemberFunc->ThisAdjustment);

      StringRef ReturnTypeName = getTypeName(MemberFunc->ReturnType);
      StringRef ClassTypeName = getTypeName(MemberFunc->ClassType);
      StringRef ArgListTypeName = getTypeName(MemberFunc->ArgListType);
      SmallString<256> TypeName(ReturnTypeName);
      TypeName.push_back(' ');
      TypeName.append(ClassTypeName);
      TypeName.append("::");
      TypeName.append(ArgListTypeName);
      Name = TypeNames.insert(TypeName).first->getKey();
      break;
    }

    case LF_METHODLIST: {
      while (!LeafData.empty()) {
        const MethodListEntry *Method;
        error(consumeObject(LeafData, Method));
        ListScope S(W, "Method");
        printMemberAttributes(Method->Attrs);
        printTypeIndex("Type", Method->Type);
        if (Method->isIntroducedVirtual()) {
          const little32_t *VFTOffsetPtr;
          error(consumeObject(LeafData, VFTOffsetPtr));
          W.printHex("VFTableOffset", *VFTOffsetPtr);
        }
      }
      break;
    }

    case LF_FUNC_ID: {
      const FuncId *Func;
      error(consumeObject(LeafData, Func));
      printTypeIndex("ParentScope", Func->ParentScope);
      printTypeIndex("FunctionType", Func->FunctionType);
      StringRef Null;
      std::tie(Name, Null) = LeafData.split('\0');
      W.printString("Name", Name);
      break;
    }

    case LF_TYPESERVER2: {
      const TypeServer2 *TypeServer;
      error(consumeObject(LeafData, TypeServer));
      W.printBinary("Signature", StringRef(TypeServer->Signature, 16));
      W.printNumber("Age", TypeServer->Age);
      Name = LeafData.split('\0').first;
      W.printString("Name", Name);
      break;
    }

    case LF_POINTER: {
      const PointerType *Ptr;
      error(consumeObject(LeafData, Ptr));
      printTypeIndex("PointeeType", Ptr->PointeeType);
      W.printHex("PointerAttributes", Ptr->Attrs);
      W.printEnum("PtrType", unsigned(Ptr->getPtrKind()),
                  makeArrayRef(PtrKindNames));
      W.printEnum("PtrMode", unsigned(Ptr->getPtrMode()),
                  makeArrayRef(PtrModeNames));
      W.printNumber("IsFlat", Ptr->isFlat());
      W.printNumber("IsConst", Ptr->isConst());
      W.printNumber("IsVolatile", Ptr->isVolatile());
      W.printNumber("IsUnaligned", Ptr->isUnaligned());

      if (Ptr->isPointerToMember()) {
        const PointerToMemberTail *PMT;
        error(consumeObject(LeafData, PMT));
        printTypeIndex("ClassType", PMT->ClassType);
        W.printEnum("Representation", PMT->Representation,
                    makeArrayRef(PtrMemberRepNames));

        StringRef PointeeName = getTypeName(Ptr->PointeeType);
        StringRef ClassName = getTypeName(PMT->ClassType);
        SmallString<256> TypeName(PointeeName);
        TypeName.push_back(' ');
        TypeName.append(ClassName);
        TypeName.append("::*");
        Name = TypeNames.insert(TypeName).first->getKey();
      } else {
        W.printBinaryBlock("TailData", LeafData);

        SmallString<256> TypeName;
        if (Ptr->isConst())
          TypeName.append("const ");
        if (Ptr->isVolatile())
          TypeName.append("volatile ");
        if (Ptr->isUnaligned())
          TypeName.append("__unaligned ");

        TypeName.append(getTypeName(Ptr->PointeeType));

        if (Ptr->getPtrMode() == PointerMode::LValueReference)
          TypeName.append("&");
        else if (Ptr->getPtrMode() == PointerMode::RValueReference)
          TypeName.append("&&");
        else if (Ptr->getPtrMode() == PointerMode::Pointer)
          TypeName.append("*");

        Name = TypeNames.insert(TypeName).first->getKey();
      }
      break;
    }

    case LF_MODIFIER: {
      const TypeModifier *Mod;
      error(consumeObject(LeafData, Mod));
      printTypeIndex("ModifiedType", Mod->ModifiedType);
      W.printFlags("Modifiers", Mod->Modifiers,
                   makeArrayRef(TypeModifierNames));

      StringRef ModifiedName = getTypeName(Mod->ModifiedType);
      SmallString<256> TypeName;
      if (Mod->Modifiers & uint16_t(ModifierOptions::Const))
        TypeName.append("const ");
      if (Mod->Modifiers & uint16_t(ModifierOptions::Volatile))
        TypeName.append("volatile ");
      if (Mod->Modifiers & uint16_t(ModifierOptions::Unaligned))
        TypeName.append("__unaligned ");
      TypeName.append(ModifiedName);
      Name = TypeNames.insert(TypeName).first->getKey();
      break;
    }

    case LF_VTSHAPE: {
      const VTableShape *Shape;
      error(consumeObject(LeafData, Shape));
      unsigned VFEntryCount = Shape->VFEntryCount;
      W.printNumber("VFEntryCount", VFEntryCount);
      // We could print out whether the methods are near or far, but in practice
      // today everything is CV_VTS_near32, so it's just noise.
      break;
    }

    case LF_UDT_SRC_LINE: {
      const UDTSrcLine *Line;
      error(consumeObject(LeafData, Line));
      printTypeIndex("UDT", Line->UDT);
      printTypeIndex("SourceFile", Line->SourceFile);
      W.printNumber("LineNumber", Line->LineNumber);
      break;
    }

    case LF_BUILDINFO: {
      const BuildInfo *Args;
      error(consumeObject(LeafData, Args));
      W.printNumber("NumArgs", Args->NumArgs);

      ListScope Arguments(W, "Arguments");
      for (uint32_t ArgI = 0; ArgI != Args->NumArgs; ++ArgI) {
        const TypeIndex *Type;
        error(consumeObject(LeafData, Type));
        printTypeIndex("ArgType", *Type);
      }
      break;
    }
    }

    if (opts::CodeViewSubsectionBytes)
      W.printBinaryBlock("LeafData", LeafData);

    CVUDTNames.push_back(Name);

    Data = RemainingData;
    // FIXME: The stream contains LF_PAD bytes that we need to ignore, but those
    // are typically included in LeafData. We may need to call skipPadding() if
    // we ever find a record that doesn't count those bytes.
  }
}

static StringRef skipPadding(StringRef Data) {
  if (Data.empty())
    return Data;
  uint8_t Leaf = Data.front();
  if (Leaf < LF_PAD0)
    return Data;
  // Leaf is greater than 0xf0. We should advance by the number of bytes in the
  // low 4 bits.
  return Data.drop_front(Leaf & 0x0F);
}

void COFFDumper::printMemberAttributes(MemberAttributes Attrs) {
  W.printEnum("AccessSpecifier", uint8_t(Attrs.getAccess()),
              makeArrayRef(MemberAccessNames));
  auto MK = Attrs.getMethodKind();
  // Data members will be vanilla. Don't try to print a method kind for them.
  if (MK != MethodKind::Vanilla)
    W.printEnum("MethodKind", unsigned(MK), makeArrayRef(MemberKindNames));
  if (Attrs.getFlags() != MethodOptions::None) {
    W.printFlags("MethodOptions", unsigned(Attrs.getFlags()),
                 makeArrayRef(MethodOptionNames));
  }
}

void COFFDumper::printCodeViewFieldList(StringRef FieldData) {
  while (!FieldData.empty()) {
    const ulittle16_t *LeafPtr;
    error(consumeObject(FieldData, LeafPtr));
    uint16_t Leaf = *LeafPtr;
    switch (Leaf) {
    default:
      W.printHex("UnknownMember", Leaf);
      // We can't advance once we hit an unknown field. The size is not encoded.
      return;

    case LF_NESTTYPE: {
      const NestedType *Nested;
      error(consumeObject(FieldData, Nested));
      DictScope S(W, "NestedType");
      printTypeIndex("Type", Nested->Type);
      StringRef Name;
      std::tie(Name, FieldData) = FieldData.split('\0');
      W.printString("Name", Name);
      break;
    }

    case LF_ONEMETHOD: {
      const OneMethod *Method;
      error(consumeObject(FieldData, Method));
      DictScope S(W, "OneMethod");
      printMemberAttributes(Method->Attrs);
      printTypeIndex("Type", Method->Type);
      // If virtual, then read the vftable offset.
      if (Method->isIntroducedVirtual()) {
        const little32_t *VFTOffsetPtr;
        error(consumeObject(FieldData, VFTOffsetPtr));
        W.printHex("VFTableOffset", *VFTOffsetPtr);
      }
      StringRef Name;
      std::tie(Name, FieldData) = FieldData.split('\0');
      W.printString("Name", Name);
      break;
    }

    case LF_METHOD: {
      const OverloadedMethod *Method;
      error(consumeObject(FieldData, Method));
      DictScope S(W, "OverloadedMethod");
      W.printHex("MethodCount", Method->MethodCount);
      W.printHex("MethodListIndex", Method->MethList.getIndex());
      StringRef Name;
      std::tie(Name, FieldData) = FieldData.split('\0');
      W.printString("Name", Name);
      break;
    }

    case LF_MEMBER: {
      const DataMember *Field;
      error(consumeObject(FieldData, Field));
      DictScope S(W, "DataMember");
      printMemberAttributes(Field->Attrs);
      printTypeIndex("Type", Field->Type);
      uint64_t FieldOffset;
      error(decodeUIntLeaf(FieldData, FieldOffset));
      W.printHex("FieldOffset", FieldOffset);
      StringRef Name;
      std::tie(Name, FieldData) = FieldData.split('\0');
      W.printString("Name", Name);
      break;
    }

    case LF_STMEMBER: {
      const StaticDataMember *Field;
      error(consumeObject(FieldData, Field));
      DictScope S(W, "StaticDataMember");
      printMemberAttributes(Field->Attrs);
      printTypeIndex("Type", Field->Type);
      StringRef Name;
      std::tie(Name, FieldData) = FieldData.split('\0');
      W.printString("Name", Name);
      break;
    }

    case LF_VFUNCTAB: {
      const VirtualFunctionPointer *VFTable;
      error(consumeObject(FieldData, VFTable));
      DictScope S(W, "VirtualFunctionPointer");
      printTypeIndex("Type", VFTable->Type);
      break;
    }

    case LF_ENUMERATE: {
      const Enumerator *Enum;
      error(consumeObject(FieldData, Enum));
      DictScope S(W, "Enumerator");
      printMemberAttributes(Enum->Attrs);
      APSInt EnumValue;
      error(decodeNumerictLeaf(FieldData, EnumValue));
      W.printNumber("EnumValue", EnumValue);
      StringRef Name;
      std::tie(Name, FieldData) = FieldData.split('\0');
      W.printString("Name", Name);
      break;
    }

    case LF_BCLASS:
    case LF_BINTERFACE: {
      const BaseClass *Base;
      error(consumeObject(FieldData, Base));
      DictScope S(W, "BaseClass");
      printMemberAttributes(Base->Attrs);
      printTypeIndex("BaseType", Base->BaseType);
      uint64_t BaseOffset;
      error(decodeUIntLeaf(FieldData, BaseOffset));
      W.printHex("BaseOffset", BaseOffset);
      break;
    }

    case LF_VBCLASS:
    case LF_IVBCLASS: {
      const VirtualBaseClass *Base;
      error(consumeObject(FieldData, Base));
      DictScope S(W, "VirtualBaseClass");
      printMemberAttributes(Base->Attrs);
      printTypeIndex("BaseType",  Base->BaseType);
      printTypeIndex("VBPtrType", Base->VBPtrType);
      uint64_t VBPtrOffset, VBTableIndex;
      error(decodeUIntLeaf(FieldData, VBPtrOffset));
      error(decodeUIntLeaf(FieldData, VBTableIndex));
      W.printHex("VBPtrOffset", VBPtrOffset);
      W.printHex("VBTableIndex", VBTableIndex);
      break;
    }
    }

    // Handle padding.
    FieldData = skipPadding(FieldData);
  }
}

void COFFDumper::printSections() {
  ListScope SectionsD(W, "Sections");
  int SectionNumber = 0;
  for (const SectionRef &Sec : Obj->sections()) {
    ++SectionNumber;
    const coff_section *Section = Obj->getCOFFSection(Sec);

    StringRef Name;
    error(Sec.getName(Name));

    DictScope D(W, "Section");
    W.printNumber("Number", SectionNumber);
    W.printBinary("Name", Name, Section->Name);
    W.printHex   ("VirtualSize", Section->VirtualSize);
    W.printHex   ("VirtualAddress", Section->VirtualAddress);
    W.printNumber("RawDataSize", Section->SizeOfRawData);
    W.printHex   ("PointerToRawData", Section->PointerToRawData);
    W.printHex   ("PointerToRelocations", Section->PointerToRelocations);
    W.printHex   ("PointerToLineNumbers", Section->PointerToLinenumbers);
    W.printNumber("RelocationCount", Section->NumberOfRelocations);
    W.printNumber("LineNumberCount", Section->NumberOfLinenumbers);
    W.printFlags ("Characteristics", Section->Characteristics,
                    makeArrayRef(ImageSectionCharacteristics),
                    COFF::SectionCharacteristics(0x00F00000));

    if (opts::SectionRelocations) {
      ListScope D(W, "Relocations");
      for (const RelocationRef &Reloc : Sec.relocations())
        printRelocation(Sec, Reloc);
    }

    if (opts::SectionSymbols) {
      ListScope D(W, "Symbols");
      for (const SymbolRef &Symbol : Obj->symbols()) {
        if (!Sec.containsSymbol(Symbol))
          continue;

        printSymbol(Symbol);
      }
    }

    if (opts::SectionData &&
        !(Section->Characteristics & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA)) {
      StringRef Data;
      error(Sec.getContents(Data));

      W.printBinaryBlock("SectionData", Data);
    }
  }
}

void COFFDumper::printRelocations() {
  ListScope D(W, "Relocations");

  int SectionNumber = 0;
  for (const SectionRef &Section : Obj->sections()) {
    ++SectionNumber;
    StringRef Name;
    error(Section.getName(Name));

    bool PrintedGroup = false;
    for (const RelocationRef &Reloc : Section.relocations()) {
      if (!PrintedGroup) {
        W.startLine() << "Section (" << SectionNumber << ") " << Name << " {\n";
        W.indent();
        PrintedGroup = true;
      }

      printRelocation(Section, Reloc);
    }

    if (PrintedGroup) {
      W.unindent();
      W.startLine() << "}\n";
    }
  }
}

void COFFDumper::printRelocation(const SectionRef &Section,
                                 const RelocationRef &Reloc, uint64_t Bias) {
  uint64_t Offset = Reloc.getOffset() - Bias;
  uint64_t RelocType = Reloc.getType();
  SmallString<32> RelocName;
  StringRef SymbolName;
  Reloc.getTypeName(RelocName);
  symbol_iterator Symbol = Reloc.getSymbol();
  if (Symbol != Obj->symbol_end()) {
    ErrorOr<StringRef> SymbolNameOrErr = Symbol->getName();
    error(SymbolNameOrErr.getError());
    SymbolName = *SymbolNameOrErr;
  }

  if (opts::ExpandRelocs) {
    DictScope Group(W, "Relocation");
    W.printHex("Offset", Offset);
    W.printNumber("Type", RelocName, RelocType);
    W.printString("Symbol", SymbolName.empty() ? "-" : SymbolName);
  } else {
    raw_ostream& OS = W.startLine();
    OS << W.hex(Offset)
       << " " << RelocName
       << " " << (SymbolName.empty() ? "-" : SymbolName)
       << "\n";
  }
}

void COFFDumper::printSymbols() {
  ListScope Group(W, "Symbols");

  for (const SymbolRef &Symbol : Obj->symbols())
    printSymbol(Symbol);
}

void COFFDumper::printDynamicSymbols() { ListScope Group(W, "DynamicSymbols"); }

static ErrorOr<StringRef>
getSectionName(const llvm::object::COFFObjectFile *Obj, int32_t SectionNumber,
               const coff_section *Section) {
  if (Section) {
    StringRef SectionName;
    if (std::error_code EC = Obj->getSectionName(Section, SectionName))
      return EC;
    return SectionName;
  }
  if (SectionNumber == llvm::COFF::IMAGE_SYM_DEBUG)
    return StringRef("IMAGE_SYM_DEBUG");
  if (SectionNumber == llvm::COFF::IMAGE_SYM_ABSOLUTE)
    return StringRef("IMAGE_SYM_ABSOLUTE");
  if (SectionNumber == llvm::COFF::IMAGE_SYM_UNDEFINED)
    return StringRef("IMAGE_SYM_UNDEFINED");
  return StringRef("");
}

void COFFDumper::printSymbol(const SymbolRef &Sym) {
  DictScope D(W, "Symbol");

  COFFSymbolRef Symbol = Obj->getCOFFSymbol(Sym);
  const coff_section *Section;
  if (std::error_code EC = Obj->getSection(Symbol.getSectionNumber(), Section)) {
    W.startLine() << "Invalid section number: " << EC.message() << "\n";
    W.flush();
    return;
  }

  StringRef SymbolName;
  if (Obj->getSymbolName(Symbol, SymbolName))
    SymbolName = "";

  StringRef SectionName = "";
  ErrorOr<StringRef> Res =
      getSectionName(Obj, Symbol.getSectionNumber(), Section);
  if (Res)
    SectionName = *Res;

  W.printString("Name", SymbolName);
  W.printNumber("Value", Symbol.getValue());
  W.printNumber("Section", SectionName, Symbol.getSectionNumber());
  W.printEnum  ("BaseType", Symbol.getBaseType(), makeArrayRef(ImageSymType));
  W.printEnum  ("ComplexType", Symbol.getComplexType(),
                                                   makeArrayRef(ImageSymDType));
  W.printEnum  ("StorageClass", Symbol.getStorageClass(),
                                                   makeArrayRef(ImageSymClass));
  W.printNumber("AuxSymbolCount", Symbol.getNumberOfAuxSymbols());

  for (uint8_t I = 0; I < Symbol.getNumberOfAuxSymbols(); ++I) {
    if (Symbol.isFunctionDefinition()) {
      const coff_aux_function_definition *Aux;
      error(getSymbolAuxData(Obj, Symbol, I, Aux));

      DictScope AS(W, "AuxFunctionDef");
      W.printNumber("TagIndex", Aux->TagIndex);
      W.printNumber("TotalSize", Aux->TotalSize);
      W.printHex("PointerToLineNumber", Aux->PointerToLinenumber);
      W.printHex("PointerToNextFunction", Aux->PointerToNextFunction);

    } else if (Symbol.isAnyUndefined()) {
      const coff_aux_weak_external *Aux;
      error(getSymbolAuxData(Obj, Symbol, I, Aux));

      ErrorOr<COFFSymbolRef> Linked = Obj->getSymbol(Aux->TagIndex);
      StringRef LinkedName;
      std::error_code EC = Linked.getError();
      if (EC || (EC = Obj->getSymbolName(*Linked, LinkedName))) {
        LinkedName = "";
        error(EC);
      }

      DictScope AS(W, "AuxWeakExternal");
      W.printNumber("Linked", LinkedName, Aux->TagIndex);
      W.printEnum  ("Search", Aux->Characteristics,
                    makeArrayRef(WeakExternalCharacteristics));

    } else if (Symbol.isFileRecord()) {
      const char *FileName;
      error(getSymbolAuxData(Obj, Symbol, I, FileName));

      DictScope AS(W, "AuxFileRecord");

      StringRef Name(FileName, Symbol.getNumberOfAuxSymbols() *
                                   Obj->getSymbolTableEntrySize());
      W.printString("FileName", Name.rtrim(StringRef("\0", 1)));
      break;
    } else if (Symbol.isSectionDefinition()) {
      const coff_aux_section_definition *Aux;
      error(getSymbolAuxData(Obj, Symbol, I, Aux));

      int32_t AuxNumber = Aux->getNumber(Symbol.isBigObj());

      DictScope AS(W, "AuxSectionDef");
      W.printNumber("Length", Aux->Length);
      W.printNumber("RelocationCount", Aux->NumberOfRelocations);
      W.printNumber("LineNumberCount", Aux->NumberOfLinenumbers);
      W.printHex("Checksum", Aux->CheckSum);
      W.printNumber("Number", AuxNumber);
      W.printEnum("Selection", Aux->Selection, makeArrayRef(ImageCOMDATSelect));

      if (Section && Section->Characteristics & COFF::IMAGE_SCN_LNK_COMDAT
          && Aux->Selection == COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE) {
        const coff_section *Assoc;
        StringRef AssocName = "";
        std::error_code EC = Obj->getSection(AuxNumber, Assoc);
        ErrorOr<StringRef> Res = getSectionName(Obj, AuxNumber, Assoc);
        if (Res)
          AssocName = *Res;
        if (!EC)
          EC = Res.getError();
        if (EC) {
          AssocName = "";
          error(EC);
        }

        W.printNumber("AssocSection", AssocName, AuxNumber);
      }
    } else if (Symbol.isCLRToken()) {
      const coff_aux_clr_token *Aux;
      error(getSymbolAuxData(Obj, Symbol, I, Aux));

      ErrorOr<COFFSymbolRef> ReferredSym =
          Obj->getSymbol(Aux->SymbolTableIndex);
      StringRef ReferredName;
      std::error_code EC = ReferredSym.getError();
      if (EC || (EC = Obj->getSymbolName(*ReferredSym, ReferredName))) {
        ReferredName = "";
        error(EC);
      }

      DictScope AS(W, "AuxCLRToken");
      W.printNumber("AuxType", Aux->AuxType);
      W.printNumber("Reserved", Aux->Reserved);
      W.printNumber("SymbolTableIndex", ReferredName, Aux->SymbolTableIndex);

    } else {
      W.startLine() << "<unhandled auxiliary record>\n";
    }
  }
}

void COFFDumper::printUnwindInfo() {
  ListScope D(W, "UnwindInformation");
  switch (Obj->getMachine()) {
  case COFF::IMAGE_FILE_MACHINE_AMD64: {
    Win64EH::Dumper Dumper(W);
    Win64EH::Dumper::SymbolResolver
    Resolver = [](const object::coff_section *Section, uint64_t Offset,
                  SymbolRef &Symbol, void *user_data) -> std::error_code {
      COFFDumper *Dumper = reinterpret_cast<COFFDumper *>(user_data);
      return Dumper->resolveSymbol(Section, Offset, Symbol);
    };
    Win64EH::Dumper::Context Ctx(*Obj, Resolver, this);
    Dumper.printData(Ctx);
    break;
  }
  case COFF::IMAGE_FILE_MACHINE_ARMNT: {
    ARM::WinEH::Decoder Decoder(W);
    Decoder.dumpProcedureData(*Obj);
    break;
  }
  default:
    W.printEnum("unsupported Image Machine", Obj->getMachine(),
                makeArrayRef(ImageFileMachineType));
    break;
  }
}

void COFFDumper::printImportedSymbols(
    iterator_range<imported_symbol_iterator> Range) {
  for (const ImportedSymbolRef &I : Range) {
    StringRef Sym;
    error(I.getSymbolName(Sym));
    uint16_t Ordinal;
    error(I.getOrdinal(Ordinal));
    W.printNumber("Symbol", Sym, Ordinal);
  }
}

void COFFDumper::printDelayImportedSymbols(
    const DelayImportDirectoryEntryRef &I,
    iterator_range<imported_symbol_iterator> Range) {
  int Index = 0;
  for (const ImportedSymbolRef &S : Range) {
    DictScope Import(W, "Import");
    StringRef Sym;
    error(S.getSymbolName(Sym));
    uint16_t Ordinal;
    error(S.getOrdinal(Ordinal));
    W.printNumber("Symbol", Sym, Ordinal);
    uint64_t Addr;
    error(I.getImportAddress(Index++, Addr));
    W.printHex("Address", Addr);
  }
}

void COFFDumper::printCOFFImports() {
  // Regular imports
  for (const ImportDirectoryEntryRef &I : Obj->import_directories()) {
    DictScope Import(W, "Import");
    StringRef Name;
    error(I.getName(Name));
    W.printString("Name", Name);
    uint32_t Addr;
    error(I.getImportLookupTableRVA(Addr));
    W.printHex("ImportLookupTableRVA", Addr);
    error(I.getImportAddressTableRVA(Addr));
    W.printHex("ImportAddressTableRVA", Addr);
    printImportedSymbols(I.imported_symbols());
  }

  // Delay imports
  for (const DelayImportDirectoryEntryRef &I : Obj->delay_import_directories()) {
    DictScope Import(W, "DelayImport");
    StringRef Name;
    error(I.getName(Name));
    W.printString("Name", Name);
    const delay_import_directory_table_entry *Table;
    error(I.getDelayImportTable(Table));
    W.printHex("Attributes", Table->Attributes);
    W.printHex("ModuleHandle", Table->ModuleHandle);
    W.printHex("ImportAddressTable", Table->DelayImportAddressTable);
    W.printHex("ImportNameTable", Table->DelayImportNameTable);
    W.printHex("BoundDelayImportTable", Table->BoundDelayImportTable);
    W.printHex("UnloadDelayImportTable", Table->UnloadDelayImportTable);
    printDelayImportedSymbols(I, I.imported_symbols());
  }
}

void COFFDumper::printCOFFExports() {
  for (const ExportDirectoryEntryRef &E : Obj->export_directories()) {
    DictScope Export(W, "Export");

    StringRef Name;
    uint32_t Ordinal, RVA;

    error(E.getSymbolName(Name));
    error(E.getOrdinal(Ordinal));
    error(E.getExportRVA(RVA));

    W.printNumber("Ordinal", Ordinal);
    W.printString("Name", Name);
    W.printHex("RVA", RVA);
  }
}

void COFFDumper::printCOFFDirectives() {
  for (const SectionRef &Section : Obj->sections()) {
    StringRef Contents;
    StringRef Name;

    error(Section.getName(Name));
    if (Name != ".drectve")
      continue;

    error(Section.getContents(Contents));

    W.printString("Directive(s)", Contents);
  }
}

static StringRef getBaseRelocTypeName(uint8_t Type) {
  switch (Type) {
  case COFF::IMAGE_REL_BASED_ABSOLUTE: return "ABSOLUTE";
  case COFF::IMAGE_REL_BASED_HIGH: return "HIGH";
  case COFF::IMAGE_REL_BASED_LOW: return "LOW";
  case COFF::IMAGE_REL_BASED_HIGHLOW: return "HIGHLOW";
  case COFF::IMAGE_REL_BASED_HIGHADJ: return "HIGHADJ";
  case COFF::IMAGE_REL_BASED_ARM_MOV32T: return "ARM_MOV32(T)";
  case COFF::IMAGE_REL_BASED_DIR64: return "DIR64";
  default: return "unknown (" + llvm::utostr(Type) + ")";
  }
}

void COFFDumper::printCOFFBaseReloc() {
  ListScope D(W, "BaseReloc");
  for (const BaseRelocRef &I : Obj->base_relocs()) {
    uint8_t Type;
    uint32_t RVA;
    error(I.getRVA(RVA));
    error(I.getType(Type));
    DictScope Import(W, "Entry");
    W.printString("Type", getBaseRelocTypeName(Type));
    W.printHex("Address", RVA);
  }
}

void COFFDumper::printStackMap() const {
  object::SectionRef StackMapSection;
  for (auto Sec : Obj->sections()) {
    StringRef Name;
    Sec.getName(Name);
    if (Name == ".llvm_stackmaps") {
      StackMapSection = Sec;
      break;
    }
  }

  if (StackMapSection == object::SectionRef())
    return;

  StringRef StackMapContents;
  StackMapSection.getContents(StackMapContents);
  ArrayRef<uint8_t> StackMapContentsArray(
      reinterpret_cast<const uint8_t*>(StackMapContents.data()),
      StackMapContents.size());

  if (Obj->isLittleEndian())
    prettyPrintStackMap(
                      llvm::outs(),
                      StackMapV1Parser<support::little>(StackMapContentsArray));
  else
    prettyPrintStackMap(llvm::outs(),
                        StackMapV1Parser<support::big>(StackMapContentsArray));
}
