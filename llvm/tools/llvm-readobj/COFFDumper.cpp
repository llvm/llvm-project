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

#include "ARMWinEHPrinter.h"
#include "CodeView.h"
#include "Error.h"
#include "ObjDumper.h"
#include "StackMapPrinter.h"
#include "Win64EHDumper.h"
#include "llvm-readobj.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/DebugInfo/CodeView/CVTypeDumper.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/Line.h"
#include "llvm/DebugInfo/CodeView/RecordSerialization.h"
#include "llvm/DebugInfo/CodeView/SymbolDeserializer.h"
#include "llvm/DebugInfo/CodeView/SymbolDumpDelegate.h"
#include "llvm/DebugInfo/CodeView/SymbolDumper.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/CodeView/TypeDumpVisitor.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeStreamMerger.h"
#include "llvm/DebugInfo/CodeView/TypeTableBuilder.h"
#include "llvm/DebugInfo/MSF/BinaryByteStream.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ScopedPrinter.h"
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
  friend class COFFObjectDumpDelegate;
  COFFDumper(const llvm::object::COFFObjectFile *Obj, ScopedPrinter &Writer)
      : ObjDumper(Writer), Obj(Obj), Writer(Writer) {}

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
  void printCOFFDebugDirectory() override;
  void printCodeViewDebugInfo() override;
  void mergeCodeViewTypes(llvm::codeview::TypeTableBuilder &CVTypes) override;
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
  StringRef getTypeName(TypeIndex Ty);
  StringRef getFileNameForFileOffset(uint32_t FileOffset);
  void printFileNameForOffset(StringRef Label, uint32_t FileOffset);
  void printTypeIndex(StringRef FieldName, TypeIndex TI) {
    // Forward to CVTypeDumper for simplicity.
    CVTypeDumper::printTypeIndex(Writer, FieldName, TI, TypeDB);
  }

  void printCodeViewSymbolsSubsection(StringRef Subsection,
                                      const SectionRef &Section,
                                      StringRef SectionContents);

  void printCodeViewFileChecksums(StringRef Subsection);

  void printCodeViewInlineeLines(StringRef Subsection);

  void printRelocatedField(StringRef Label, const coff_section *Sec,
                           uint32_t RelocOffset, uint32_t Offset,
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

  ScopedPrinter &Writer;
  TypeDatabase TypeDB;
};

class COFFObjectDumpDelegate : public SymbolDumpDelegate {
public:
  COFFObjectDumpDelegate(COFFDumper &CD, const SectionRef &SR,
                         const COFFObjectFile *Obj, StringRef SectionContents)
      : CD(CD), SR(SR), SectionContents(SectionContents) {
    Sec = Obj->getCOFFSection(SR);
  }

  uint32_t getRecordOffset(BinaryStreamReader Reader) override {
    ArrayRef<uint8_t> Data;
    if (auto EC = Reader.readLongestContiguousChunk(Data)) {
      llvm::consumeError(std::move(EC));
      return 0;
    }
    return Data.data() - SectionContents.bytes_begin();
  }

  void printRelocatedField(StringRef Label, uint32_t RelocOffset,
                           uint32_t Offset, StringRef *RelocSym) override {
    CD.printRelocatedField(Label, Sec, RelocOffset, Offset, RelocSym);
  }

  void printBinaryBlockWithRelocs(StringRef Label,
                                  ArrayRef<uint8_t> Block) override {
    StringRef SBlock(reinterpret_cast<const char *>(Block.data()),
                     Block.size());
    if (opts::CodeViewSubsectionBytes)
      CD.printBinaryBlockWithRelocs(Label, SR, SectionContents, SBlock);
  }

  StringRef getFileNameForFileOffset(uint32_t FileOffset) override {
    return CD.getFileNameForFileOffset(FileOffset);
  }

  StringRef getStringTable() override { return CD.CVStringTable; }

private:
  COFFDumper &CD;
  const SectionRef &SR;
  const coff_section *Sec;
  StringRef SectionContents;
};

} // end namespace

namespace llvm {

std::error_code createCOFFDumper(const object::ObjectFile *Obj,
                                 ScopedPrinter &Writer,
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
  auto SymI = Obj->symbol_end();
  for (const auto &Relocation : Relocations) {
    uint64_t RelocationOffset = Relocation.getOffset();

    if (RelocationOffset == Offset) {
      SymI = Relocation.getSymbol();
      break;
    }
  }
  if (SymI == Obj->symbol_end())
    return readobj_error::unknown_symbol;
  Sym = *SymI;
  return readobj_error::success;
}

// Given a section and an offset into this section the function returns the name
// of the symbol used for the relocation at the offset.
std::error_code COFFDumper::resolveSymbolName(const coff_section *Section,
                                              uint64_t Offset,
                                              StringRef &Name) {
  SymbolRef Symbol;
  if (std::error_code EC = resolveSymbol(Section, Offset, Symbol))
    return EC;
  Expected<StringRef> NameOrErr = Symbol.getName();
  if (!NameOrErr)
    return errorToErrorCode(NameOrErr.takeError());
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
                                     uint32_t RelocOffset, uint32_t Offset,
                                     StringRef *RelocSym) {
  StringRef SymStorage;
  StringRef &Symbol = RelocSym ? *RelocSym : SymStorage;
  if (!resolveSymbolName(Sec, RelocOffset, Symbol))
    W.printSymbolOffset(Label, Symbol, Offset);
  else
    W.printHex(Label, RelocOffset);
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

  W.flush();
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

static const EnumEntry<COFF::DebugType> ImageDebugType[] = {
  { "Unknown"    , COFF::IMAGE_DEBUG_TYPE_UNKNOWN       },
  { "COFF"       , COFF::IMAGE_DEBUG_TYPE_COFF          },
  { "CodeView"   , COFF::IMAGE_DEBUG_TYPE_CODEVIEW      },
  { "FPO"        , COFF::IMAGE_DEBUG_TYPE_FPO           },
  { "Misc"       , COFF::IMAGE_DEBUG_TYPE_MISC          },
  { "Exception"  , COFF::IMAGE_DEBUG_TYPE_EXCEPTION     },
  { "Fixup"      , COFF::IMAGE_DEBUG_TYPE_FIXUP         },
  { "OmapToSrc"  , COFF::IMAGE_DEBUG_TYPE_OMAP_TO_SRC   },
  { "OmapFromSrc", COFF::IMAGE_DEBUG_TYPE_OMAP_FROM_SRC },
  { "Borland"    , COFF::IMAGE_DEBUG_TYPE_BORLAND       },
  { "Reserved10" , COFF::IMAGE_DEBUG_TYPE_RESERVED10    },
  { "CLSID"      , COFF::IMAGE_DEBUG_TYPE_CLSID         },
  { "VCFeature"  , COFF::IMAGE_DEBUG_TYPE_VC_FEATURE    },
  { "POGO"       , COFF::IMAGE_DEBUG_TYPE_POGO          },
  { "ILTCG"      , COFF::IMAGE_DEBUG_TYPE_ILTCG         },
  { "MPX"        , COFF::IMAGE_DEBUG_TYPE_MPX           },
  { "Repro"      , COFF::IMAGE_DEBUG_TYPE_REPRO         },
};

static const EnumEntry<COFF::WeakExternalCharacteristics>
WeakExternalCharacteristics[] = {
  { "NoLibrary", COFF::IMAGE_WEAK_EXTERN_SEARCH_NOLIBRARY },
  { "Library"  , COFF::IMAGE_WEAK_EXTERN_SEARCH_LIBRARY   },
  { "Alias"    , COFF::IMAGE_WEAK_EXTERN_SEARCH_ALIAS     }
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

static const EnumEntry<uint32_t> FrameDataFlags[] = {
    LLVM_READOBJ_ENUM_ENT(FrameData, HasSEH),
    LLVM_READOBJ_ENUM_ENT(FrameData, HasEH),
    LLVM_READOBJ_ENUM_ENT(FrameData, IsFunctionStart),
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

    for (uint32_t i = 0; i < Hdr->NumberOfRvaAndSize; ++i)
      printDataDirectory(i, directory[i]);
  }
}

void COFFDumper::printCOFFDebugDirectory() {
  ListScope LS(W, "DebugDirectory");
  for (const debug_directory &D : Obj->debug_directories()) {
    char FormattedTime[20] = {};
    time_t TDS = D.TimeDateStamp;
    strftime(FormattedTime, 20, "%Y-%m-%d %H:%M:%S", gmtime(&TDS));
    DictScope S(W, "DebugEntry");
    W.printHex("Characteristics", D.Characteristics);
    W.printHex("TimeDateStamp", FormattedTime, D.TimeDateStamp);
    W.printHex("MajorVersion", D.MajorVersion);
    W.printHex("MinorVersion", D.MinorVersion);
    W.printEnum("Type", D.Type, makeArrayRef(ImageDebugType));
    W.printHex("SizeOfData", D.SizeOfData);
    W.printHex("AddressOfRawData", D.AddressOfRawData);
    W.printHex("PointerToRawData", D.PointerToRawData);
    if (D.Type == COFF::IMAGE_DEBUG_TYPE_CODEVIEW) {
      const codeview::DebugInfo *DebugInfo;
      StringRef PDBFileName;
      error(Obj->getDebugPDBInfo(&D, DebugInfo, PDBFileName));
      DictScope PDBScope(W, "PDBInfo");
      W.printHex("PDBSignature", DebugInfo->Signature.CVSignature);
      if (DebugInfo->Signature.CVSignature == OMF::Signature::PDB70) {
        W.printBinary("PDBGUID", makeArrayRef(DebugInfo->PDB70.Signature));
        W.printNumber("PDBAge", DebugInfo->PDB70.Age);
        W.printString("PDBFileName", PDBFileName);
      }
    } else {
      // FIXME: Type values of 12 and 13 are commonly observed but are not in
      // the documented type enum.  Figure out what they mean.
      ArrayRef<uint8_t> RawData;
      error(
          Obj->getRvaAndSizeAsBytes(D.AddressOfRawData, D.SizeOfData, RawData));
      W.printBinaryBlock("RawData", RawData);
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

void COFFDumper::initializeFileAndStringTables(StringRef Data) {
  while (!Data.empty() && (CVFileChecksumTable.data() == nullptr ||
                           CVStringTable.data() == nullptr)) {
    // The section consists of a number of subsection in the following format:
    // |SubSectionType|SubSectionSize|Contents...|
    uint32_t SubType, SubSectionSize;
    error(consume(Data, SubType));
    error(consume(Data, SubSectionSize));
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
    uint32_t PaddedSize = alignTo(SubSectionSize, 4);
    if (PaddedSize > Data.size())
      error(object_error::parse_failed);
    Data = Data.drop_front(PaddedSize);
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
  error(consume(Data, Magic));
  W.printHex("Magic", Magic);
  if (Magic != COFF::DEBUG_SECTION_MAGIC)
    return error(object_error::parse_failed);

  initializeFileAndStringTables(Data);

  // TODO: Convert this over to using ModuleSubstreamVisitor.
  while (!Data.empty()) {
    // The section consists of a number of subsection in the following format:
    // |SubSectionType|SubSectionSize|Contents...|
    uint32_t SubType, SubSectionSize;
    error(consume(Data, SubType));
    error(consume(Data, SubSectionSize));

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
    if (NextOffset > SectionContents.size())
      return error(object_error::parse_failed);
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
      BinaryByteStream S(Contents, llvm::support::little);
      BinaryStreamReader SR(S);
      StringRef CodePtr;
      error(SR.readFixedString(CodePtr, 4));
      StringRef LinkageName;
      error(resolveSymbolName(Obj->getCOFFSection(Section), SectionContents,
                              CodePtr.data(), LinkageName));
      W.printString("LinkageName", LinkageName);

      // To find the active frame description, search this array for the
      // smallest PC range that includes the current PC.
      while (!SR.empty()) {
        const FrameData *FD;
        error(SR.readObject(FD));

        if (FD->FrameFunc >= CVStringTable.size())
          error(object_error::parse_failed);

        StringRef FrameFunc =
            CVStringTable.drop_front(FD->FrameFunc).split('\0').first;

        DictScope S(W, "FrameData");
        W.printHex("RvaStart", FD->RvaStart);
        W.printHex("CodeSize", FD->CodeSize);
        W.printHex("LocalSize", FD->LocalSize);
        W.printHex("ParamsSize", FD->ParamsSize);
        W.printHex("MaxStackSize", FD->MaxStackSize);
        W.printString("FrameFunc", FrameFunc);
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

void COFFDumper::printCodeViewSymbolsSubsection(StringRef Subsection,
                                                const SectionRef &Section,
                                                StringRef SectionContents) {
  ArrayRef<uint8_t> BinaryData(Subsection.bytes_begin(),
                               Subsection.bytes_end());
  auto CODD = llvm::make_unique<COFFObjectDumpDelegate>(*this, Section, Obj,
                                                        SectionContents);

  CVSymbolDumper CVSD(W, TypeDB, std::move(CODD),
                      opts::CodeViewSubsectionBytes);
  BinaryByteStream Stream(BinaryData, llvm::support::little);
  CVSymbolArray Symbols;
  BinaryStreamReader Reader(Stream);
  if (auto EC = Reader.readArray(Symbols, Reader.getLength())) {
    consumeError(std::move(EC));
    W.flush();
    error(object_error::parse_failed);
  }

  if (auto EC = CVSD.dump(Symbols)) {
    W.flush();
    error(std::move(EC));
  }
  W.flush();
}

void COFFDumper::printCodeViewFileChecksums(StringRef Subsection) {
  BinaryByteStream S(Subsection, llvm::support::little);
  BinaryStreamReader SR(S);
  while (!SR.empty()) {
    DictScope S(W, "FileChecksum");
    const FileChecksum *FC;
    error(SR.readObject(FC));
    if (FC->FileNameOffset >= CVStringTable.size())
      error(object_error::parse_failed);
    StringRef Filename =
        CVStringTable.drop_front(FC->FileNameOffset).split('\0').first;
    W.printHex("Filename", Filename, FC->FileNameOffset);
    W.printHex("ChecksumSize", FC->ChecksumSize);
    W.printEnum("ChecksumKind", uint8_t(FC->ChecksumKind),
                makeArrayRef(FileChecksumKindNames));
    if (FC->ChecksumSize >= SR.bytesRemaining())
      error(object_error::parse_failed);
    ArrayRef<uint8_t> ChecksumBytes;
    error(SR.readBytes(ChecksumBytes, FC->ChecksumSize));
    W.printBinary("ChecksumBytes", ChecksumBytes);
    unsigned PaddedSize = alignTo(FC->ChecksumSize + sizeof(FileChecksum), 4) -
                          sizeof(FileChecksum);
    PaddedSize -= ChecksumBytes.size();
    if (PaddedSize > SR.bytesRemaining())
      error(object_error::parse_failed);
    error(SR.skip(PaddedSize));
  }
}

void COFFDumper::printCodeViewInlineeLines(StringRef Subsection) {
  BinaryByteStream S(Subsection, llvm::support::little);
  BinaryStreamReader SR(S);
  uint32_t Signature;
  error(SR.readInteger(Signature));
  bool HasExtraFiles = Signature == unsigned(InlineeLinesSignature::ExtraFiles);

  while (!SR.empty()) {
    const InlineeSourceLine *ISL;
    error(SR.readObject(ISL));
    DictScope S(W, "InlineeSourceLine");
    printTypeIndex("Inlinee", ISL->Inlinee);
    printFileNameForOffset("FileID", ISL->FileID);
    W.printNumber("SourceLineNum", ISL->SourceLineNum);

    if (HasExtraFiles) {
      uint32_t ExtraFileCount;
      error(SR.readInteger(ExtraFileCount));
      W.printNumber("ExtraFileCount", ExtraFileCount);
      ListScope ExtraFiles(W, "ExtraFiles");
      for (unsigned I = 0; I < ExtraFileCount; ++I) {
        uint32_t FileID;
        error(SR.readInteger(FileID));
        printFileNameForOffset("FileID", FileID);
      }
    }
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
  error(consume(Data, StringOffset));

  // Check if the string table offset is valid.
  if (StringOffset >= CVStringTable.size())
    error(object_error::parse_failed);

  // Return the null-terminated string.
  return CVStringTable.drop_front(StringOffset).split('\0').first;
}

void COFFDumper::printFileNameForOffset(StringRef Label, uint32_t FileOffset) {
  W.printHex(Label, getFileNameForFileOffset(FileOffset), FileOffset);
}

void COFFDumper::mergeCodeViewTypes(TypeTableBuilder &CVTypes) {
  for (const SectionRef &S : Obj->sections()) {
    StringRef SectionName;
    error(S.getName(SectionName));
    if (SectionName == ".debug$T") {
      StringRef Data;
      error(S.getContents(Data));
      uint32_t Magic;
      error(consume(Data, Magic));
      if (Magic != 4)
        error(object_error::parse_failed);
      ArrayRef<uint8_t> Bytes(reinterpret_cast<const uint8_t *>(Data.data()),
                              Data.size());
      BinaryByteStream Stream(Bytes, llvm::support::little);
      CVTypeArray Types;
      BinaryStreamReader Reader(Stream);
      if (auto EC = Reader.readArray(Types, Reader.getLength())) {
        consumeError(std::move(EC));
        W.flush();
        error(object_error::parse_failed);
      }

      if (auto EC = mergeTypeStreams(CVTypes, nullptr, Types)) {
        consumeError(std::move(EC));
        return error(object_error::parse_failed);
      }
    }
  }
}

void COFFDumper::printCodeViewTypeSection(StringRef SectionName,
                                          const SectionRef &Section) {
  ListScope D(W, "CodeViewTypes");
  W.printNumber("Section", SectionName, Obj->getSectionID(Section));

  StringRef Data;
  error(Section.getContents(Data));
  if (opts::CodeViewSubsectionBytes)
    W.printBinaryBlock("Data", Data);

  uint32_t Magic;
  error(consume(Data, Magic));
  W.printHex("Magic", Magic);
  if (Magic != COFF::DEBUG_SECTION_MAGIC)
    return error(object_error::parse_failed);

  CVTypeDumper CVTD(TypeDB);
  TypeDumpVisitor TDV(TypeDB, &W, opts::CodeViewSubsectionBytes);
  if (auto EC = CVTD.dump({Data.bytes_begin(), Data.bytes_end()}, TDV)) {
    W.flush();
    error(llvm::errorToErrorCode(std::move(EC)));
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
    Expected<StringRef> SymbolNameOrErr = Symbol->getName();
    error(errorToErrorCode(SymbolNameOrErr.takeError()));
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
                      StackMapV2Parser<support::little>(StackMapContentsArray));
  else
    prettyPrintStackMap(llvm::outs(),
                        StackMapV2Parser<support::big>(StackMapContentsArray));
}

void llvm::dumpCodeViewMergedTypes(ScopedPrinter &Writer,
                                   llvm::codeview::TypeTableBuilder &CVTypes) {
  // Flatten it first, then run our dumper on it.
  ListScope S(Writer, "MergedTypeStream");
  SmallString<0> Buf;
  CVTypes.ForEachRecord([&](TypeIndex TI, ArrayRef<uint8_t> Record) {
    Buf.append(Record.begin(), Record.end());
  });

  TypeDatabase TypeDB;
  CVTypeDumper CVTD(TypeDB);
  TypeDumpVisitor TDV(TypeDB, &Writer, opts::CodeViewSubsectionBytes);
  if (auto EC =
          CVTD.dump({Buf.str().bytes_begin(), Buf.str().bytes_end()}, TDV)) {
    Writer.flush();
    error(llvm::errorToErrorCode(std::move(EC)));
  }
}
