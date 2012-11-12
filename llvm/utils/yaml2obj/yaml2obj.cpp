//===- yaml2obj - Convert YAML to a binary object file --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program takes a YAML description of an object file and outputs the
// binary equivalent.
//
// This is used for writing tests that require binary files.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/system_error.h"
#include "llvm/Support/YAMLParser.h"

#include <vector>

using namespace llvm;

static cl::opt<std::string>
  Input(cl::Positional, cl::desc("<input>"), cl::init("-"));

template<class T>
typename llvm::enable_if_c<std::numeric_limits<T>::is_integer, bool>::type
getAs(const llvm::yaml::ScalarNode *SN, T &Result) {
  SmallString<4> Storage;
  StringRef Value = SN->getValue(Storage);
  if (Value.getAsInteger(0, Result))
    return false;
  return true;
}

// Given a container with begin and end with ::value_type of a character type.
// Iterate through pairs of characters in the the set of [a-fA-F0-9] ignoring
// all other characters.
struct hex_pair_iterator {
  StringRef::const_iterator Current, End;
  typedef SmallVector<char, 2> value_type;
  value_type Pair;
  bool IsDone;

  hex_pair_iterator(StringRef C)
    : Current(C.begin()), End(C.end()), IsDone(false) {
    // Initalize Pair.
    ++*this;
  }

  // End iterator.
  hex_pair_iterator() : Current(), End(), IsDone(true) {}

  value_type operator *() const {
    return Pair;
  }

  hex_pair_iterator operator ++() {
    // We're at the end of the input.
    if (Current == End) {
      IsDone = true;
      return *this;
    }
    Pair = value_type();
    for (; Current != End && Pair.size() != 2; ++Current) {
      // Is a valid hex digit.
      if ((*Current >= '0' && *Current <= '9') ||
          (*Current >= 'a' && *Current <= 'f') ||
          (*Current >= 'A' && *Current <= 'F'))
        Pair.push_back(*Current);
    }
    // Hit the end without getting 2 hex digits. Pair is invalid.
    if (Pair.size() != 2)
      IsDone = true;
    return *this;
  }

  bool operator ==(const hex_pair_iterator Other) {
    return (IsDone == Other.IsDone) ||
           (Current == Other.Current && End == Other.End);
  }

  bool operator !=(const hex_pair_iterator Other) {
    return !(*this == Other);
  }
};

template <class ContainerOut>
static bool hexStringToByteArray(StringRef Str, ContainerOut &Out) {
  for (hex_pair_iterator I(Str), E; I != E; ++I) {
    typename hex_pair_iterator::value_type Pair = *I;
    typename ContainerOut::value_type Byte;
    if (StringRef(Pair.data(), 2).getAsInteger(16, Byte))
      return false;
    Out.push_back(Byte);
  }
  return true;
}

/// This parses a yaml stream that represents a COFF object file.
/// See docs/yaml2obj for the yaml scheema.
struct COFFParser {
  COFFParser(yaml::Stream &Input) : YS(Input) {
    std::memset(&Header, 0, sizeof(Header));
    // A COFF string table always starts with a 4 byte size field. Offsets into
    // it include this size, so allocate it now.
    StringTable.append(4, 0);
  }

  bool parseHeader(yaml::Node *HeaderN) {
    yaml::MappingNode *MN = dyn_cast<yaml::MappingNode>(HeaderN);
    if (!MN) {
      YS.printError(HeaderN, "header's value must be a mapping node");
      return false;
    }
    for (yaml::MappingNode::iterator i = MN->begin(), e = MN->end();
                                     i != e; ++i) {
      yaml::ScalarNode *Key = dyn_cast<yaml::ScalarNode>(i->getKey());
      if (!Key) {
        YS.printError(i->getKey(), "Keys must be scalar values");
        return false;
      }
      SmallString<32> Storage;
      StringRef KeyValue = Key->getValue(Storage);
      if (KeyValue == "Characteristics") {
        if (!parseHeaderCharacteristics(i->getValue()))
          return false;
      } else {
        yaml::ScalarNode *Value = dyn_cast<yaml::ScalarNode>(i->getValue());
        if (!Value) {
          YS.printError(Value,
            Twine(KeyValue) + " must be a scalar value");
          return false;
        }
        if (KeyValue == "Machine") {
          uint16_t Machine = COFF::MT_Invalid;
          if (!getAs(Value, Machine)) {
            // It's not a raw number, try matching the string.
            StringRef ValueValue = Value->getValue(Storage);
            Machine = StringSwitch<COFF::MachineTypes>(ValueValue)
              .Case( "IMAGE_FILE_MACHINE_UNKNOWN"
                   , COFF::IMAGE_FILE_MACHINE_UNKNOWN)
              .Case( "IMAGE_FILE_MACHINE_AM33"
                   , COFF::IMAGE_FILE_MACHINE_AM33)
              .Case( "IMAGE_FILE_MACHINE_AMD64"
                   , COFF::IMAGE_FILE_MACHINE_AMD64)
              .Case( "IMAGE_FILE_MACHINE_ARM"
                   , COFF::IMAGE_FILE_MACHINE_ARM)
              .Case( "IMAGE_FILE_MACHINE_ARMV7"
                   , COFF::IMAGE_FILE_MACHINE_ARMV7)
              .Case( "IMAGE_FILE_MACHINE_EBC"
                   , COFF::IMAGE_FILE_MACHINE_EBC)
              .Case( "IMAGE_FILE_MACHINE_I386"
                   , COFF::IMAGE_FILE_MACHINE_I386)
              .Case( "IMAGE_FILE_MACHINE_IA64"
                   , COFF::IMAGE_FILE_MACHINE_IA64)
              .Case( "IMAGE_FILE_MACHINE_M32R"
                   , COFF::IMAGE_FILE_MACHINE_M32R)
              .Case( "IMAGE_FILE_MACHINE_MIPS16"
                   , COFF::IMAGE_FILE_MACHINE_MIPS16)
              .Case( "IMAGE_FILE_MACHINE_MIPSFPU"
                   , COFF::IMAGE_FILE_MACHINE_MIPSFPU)
              .Case( "IMAGE_FILE_MACHINE_MIPSFPU16"
                   , COFF::IMAGE_FILE_MACHINE_MIPSFPU16)
              .Case( "IMAGE_FILE_MACHINE_POWERPC"
                   , COFF::IMAGE_FILE_MACHINE_POWERPC)
              .Case( "IMAGE_FILE_MACHINE_POWERPCFP"
                   , COFF::IMAGE_FILE_MACHINE_POWERPCFP)
              .Case( "IMAGE_FILE_MACHINE_R4000"
                   , COFF::IMAGE_FILE_MACHINE_R4000)
              .Case( "IMAGE_FILE_MACHINE_SH3"
                   , COFF::IMAGE_FILE_MACHINE_SH3)
              .Case( "IMAGE_FILE_MACHINE_SH3DSP"
                   , COFF::IMAGE_FILE_MACHINE_SH3DSP)
              .Case( "IMAGE_FILE_MACHINE_SH4"
                   , COFF::IMAGE_FILE_MACHINE_SH4)
              .Case( "IMAGE_FILE_MACHINE_SH5"
                   , COFF::IMAGE_FILE_MACHINE_SH5)
              .Case( "IMAGE_FILE_MACHINE_THUMB"
                   , COFF::IMAGE_FILE_MACHINE_THUMB)
              .Case( "IMAGE_FILE_MACHINE_WCEMIPSV2"
                   , COFF::IMAGE_FILE_MACHINE_WCEMIPSV2)
              .Default(COFF::MT_Invalid);
            if (Machine == COFF::MT_Invalid) {
              YS.printError(Value, "Invalid value for Machine");
              return false;
            }
          }
          Header.Machine = Machine;
        } else if (KeyValue == "NumberOfSections") {
          if (!getAs(Value, Header.NumberOfSections)) {
              YS.printError(Value, "Invalid value for NumberOfSections");
              return false;
          }
        } else if (KeyValue == "TimeDateStamp") {
          if (!getAs(Value, Header.TimeDateStamp)) {
              YS.printError(Value, "Invalid value for TimeDateStamp");
              return false;
          }
        } else if (KeyValue == "PointerToSymbolTable") {
          if (!getAs(Value, Header.PointerToSymbolTable)) {
              YS.printError(Value, "Invalid value for PointerToSymbolTable");
              return false;
          }
        } else if (KeyValue == "NumberOfSymbols") {
          if (!getAs(Value, Header.NumberOfSymbols)) {
              YS.printError(Value, "Invalid value for NumberOfSymbols");
              return false;
          }
        } else if (KeyValue == "SizeOfOptionalHeader") {
          if (!getAs(Value, Header.SizeOfOptionalHeader)) {
              YS.printError(Value, "Invalid value for SizeOfOptionalHeader");
              return false;
          }
        } else {
          YS.printError(Key, "Unrecognized key in header");
          return false;
        }
      }
    }
    return true;
  }

  bool parseHeaderCharacteristics(yaml::Node *Characteristics) {
    yaml::ScalarNode *Value = dyn_cast<yaml::ScalarNode>(Characteristics);
    yaml::SequenceNode *SeqValue
      = dyn_cast<yaml::SequenceNode>(Characteristics);
    if (!Value && !SeqValue) {
      YS.printError(Characteristics,
        "Characteristics must either be a number or sequence");
      return false;
    }
    if (Value) {
      if (!getAs(Value, Header.Characteristics)) {
        YS.printError(Value, "Invalid value for Characteristics");
        return false;
      }
    } else {
      for (yaml::SequenceNode::iterator ci = SeqValue->begin(),
                                        ce = SeqValue->end();
                                        ci != ce; ++ci) {
        yaml::ScalarNode *CharValue = dyn_cast<yaml::ScalarNode>(&*ci);
        if (!CharValue) {
          YS.printError(CharValue,
            "Characteristics must be scalar values");
          return false;
        }
        SmallString<32> Storage;
        StringRef Char = CharValue->getValue(Storage);
        uint16_t Characteristic = StringSwitch<COFF::Characteristics>(Char)
          .Case( "IMAGE_FILE_RELOCS_STRIPPED"
                , COFF::IMAGE_FILE_RELOCS_STRIPPED)
          .Case( "IMAGE_FILE_EXECUTABLE_IMAGE"
                , COFF::IMAGE_FILE_EXECUTABLE_IMAGE)
          .Case( "IMAGE_FILE_LINE_NUMS_STRIPPED"
                , COFF::IMAGE_FILE_LINE_NUMS_STRIPPED)
          .Case( "IMAGE_FILE_LOCAL_SYMS_STRIPPED"
                , COFF::IMAGE_FILE_LOCAL_SYMS_STRIPPED)
          .Case( "IMAGE_FILE_AGGRESSIVE_WS_TRIM"
                , COFF::IMAGE_FILE_AGGRESSIVE_WS_TRIM)
          .Case( "IMAGE_FILE_LARGE_ADDRESS_AWARE"
                , COFF::IMAGE_FILE_LARGE_ADDRESS_AWARE)
          .Case( "IMAGE_FILE_BYTES_REVERSED_LO"
                , COFF::IMAGE_FILE_BYTES_REVERSED_LO)
          .Case( "IMAGE_FILE_32BIT_MACHINE"
                , COFF::IMAGE_FILE_32BIT_MACHINE)
          .Case( "IMAGE_FILE_DEBUG_STRIPPED"
                , COFF::IMAGE_FILE_DEBUG_STRIPPED)
          .Case( "IMAGE_FILE_REMOVABLE_RUN_FROM_SWAP"
                , COFF::IMAGE_FILE_REMOVABLE_RUN_FROM_SWAP)
          .Case( "IMAGE_FILE_SYSTEM"
                , COFF::IMAGE_FILE_SYSTEM)
          .Case( "IMAGE_FILE_DLL"
                , COFF::IMAGE_FILE_DLL)
          .Case( "IMAGE_FILE_UP_SYSTEM_ONLY"
                , COFF::IMAGE_FILE_UP_SYSTEM_ONLY)
          .Default(COFF::C_Invalid);
        if (Characteristic == COFF::C_Invalid) {
          // TODO: Typo-correct.
          YS.printError(CharValue,
            "Invalid value for Characteristic");
          return false;
        }
        Header.Characteristics |= Characteristic;
      }
    }
    return true;
  }

  bool parseSections(yaml::Node *SectionsN) {
    yaml::SequenceNode *SN = dyn_cast<yaml::SequenceNode>(SectionsN);
    if (!SN) {
      YS.printError(SectionsN, "Sections must be a sequence");
      return false;
    }
    for (yaml::SequenceNode::iterator i = SN->begin(), e = SN->end();
                                      i != e; ++i) {
      Section Sec;
      std::memset(&Sec.Header, 0, sizeof(Sec.Header));
      yaml::MappingNode *SecMap = dyn_cast<yaml::MappingNode>(&*i);
      if (!SecMap) {
        YS.printError(&*i, "Section entry must be a map");
        return false;
      }
      for (yaml::MappingNode::iterator si = SecMap->begin(), se = SecMap->end();
                                       si != se; ++si) {
        yaml::ScalarNode *Key = dyn_cast<yaml::ScalarNode>(si->getKey());
        if (!Key) {
          YS.printError(si->getKey(), "Keys must be scalar values");
          return false;
        }
        SmallString<32> Storage;
        StringRef KeyValue = Key->getValue(Storage);

        yaml::ScalarNode *Value = dyn_cast<yaml::ScalarNode>(si->getValue());
        if (KeyValue == "Name") {
          // If the name is less than 8 bytes, store it in place, otherwise
          // store it in the string table.
          StringRef Name = Value->getValue(Storage);
          std::fill_n(Sec.Header.Name, unsigned(COFF::NameSize), 0);
          if (Name.size() <= COFF::NameSize) {
            std::copy(Name.begin(), Name.end(), Sec.Header.Name);
          } else {
            // Add string to the string table and format the index for output.
            unsigned Index = getStringIndex(Name);
            std::string str = utostr(Index);
            if (str.size() > 7) {
              YS.printError(Value, "String table got too large");
              return false;
            }
            Sec.Header.Name[0] = '/';
            std::copy(str.begin(), str.end(), Sec.Header.Name + 1);
          }
        } else if (KeyValue == "VirtualSize") {
          if (!getAs(Value, Sec.Header.VirtualSize)) {
            YS.printError(Value, "Invalid value for VirtualSize");
            return false;
          }
        } else if (KeyValue == "VirtualAddress") {
          if (!getAs(Value, Sec.Header.VirtualAddress)) {
            YS.printError(Value, "Invalid value for VirtualAddress");
            return false;
          }
        } else if (KeyValue == "SizeOfRawData") {
          if (!getAs(Value, Sec.Header.SizeOfRawData)) {
            YS.printError(Value, "Invalid value for SizeOfRawData");
            return false;
          }
        } else if (KeyValue == "PointerToRawData") {
          if (!getAs(Value, Sec.Header.PointerToRawData)) {
            YS.printError(Value, "Invalid value for PointerToRawData");
            return false;
          }
        } else if (KeyValue == "PointerToRelocations") {
          if (!getAs(Value, Sec.Header.PointerToRelocations)) {
            YS.printError(Value, "Invalid value for PointerToRelocations");
            return false;
          }
        } else if (KeyValue == "PointerToLineNumbers") {
          if (!getAs(Value, Sec.Header.PointerToLineNumbers)) {
            YS.printError(Value, "Invalid value for PointerToLineNumbers");
            return false;
          }
        } else if (KeyValue == "NumberOfRelocations") {
          if (!getAs(Value, Sec.Header.NumberOfRelocations)) {
            YS.printError(Value, "Invalid value for NumberOfRelocations");
            return false;
          }
        } else if (KeyValue == "NumberOfLineNumbers") {
          if (!getAs(Value, Sec.Header.NumberOfLineNumbers)) {
            YS.printError(Value, "Invalid value for NumberOfLineNumbers");
            return false;
          }
        } else if (KeyValue == "Characteristics") {
          yaml::SequenceNode *SeqValue
            = dyn_cast<yaml::SequenceNode>(si->getValue());
          if (!Value && !SeqValue) {
            YS.printError(si->getValue(),
              "Characteristics must either be a number or sequence");
            return false;
          }
          if (Value) {
            if (!getAs(Value, Sec.Header.Characteristics)) {
              YS.printError(Value, "Invalid value for Characteristics");
              return false;
            }
          } else {
            for (yaml::SequenceNode::iterator ci = SeqValue->begin(),
                                              ce = SeqValue->end();
                                              ci != ce; ++ci) {
              yaml::ScalarNode *CharValue = dyn_cast<yaml::ScalarNode>(&*ci);
              if (!CharValue) {
                YS.printError(CharValue, "Invalid value for Characteristics");
                return false;
              }
              StringRef Char = CharValue->getValue(Storage);
              uint32_t Characteristic =
                StringSwitch<COFF::SectionCharacteristics>(Char)
                .Case( "IMAGE_SCN_TYPE_NO_PAD"
                     , COFF::IMAGE_SCN_TYPE_NO_PAD)
                .Case( "IMAGE_SCN_CNT_CODE"
                     , COFF::IMAGE_SCN_CNT_CODE)
                .Case( "IMAGE_SCN_CNT_INITIALIZED_DATA"
                     , COFF::IMAGE_SCN_CNT_INITIALIZED_DATA)
                .Case( "IMAGE_SCN_CNT_UNINITIALIZED_DATA"
                     , COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA)
                .Case( "IMAGE_SCN_LNK_OTHER"
                     , COFF::IMAGE_SCN_LNK_OTHER)
                .Case( "IMAGE_SCN_LNK_INFO"
                     , COFF::IMAGE_SCN_LNK_INFO)
                .Case( "IMAGE_SCN_LNK_REMOVE"
                     , COFF::IMAGE_SCN_LNK_REMOVE)
                .Case( "IMAGE_SCN_LNK_COMDAT"
                     , COFF::IMAGE_SCN_LNK_COMDAT)
                .Case( "IMAGE_SCN_GPREL"
                     , COFF::IMAGE_SCN_GPREL)
                .Case( "IMAGE_SCN_MEM_PURGEABLE"
                     , COFF::IMAGE_SCN_MEM_PURGEABLE)
                .Case( "IMAGE_SCN_MEM_16BIT"
                     , COFF::IMAGE_SCN_MEM_16BIT)
                .Case( "IMAGE_SCN_MEM_LOCKED"
                     , COFF::IMAGE_SCN_MEM_LOCKED)
                .Case( "IMAGE_SCN_MEM_PRELOAD"
                     , COFF::IMAGE_SCN_MEM_PRELOAD)
                .Case( "IMAGE_SCN_ALIGN_1BYTES"
                     , COFF::IMAGE_SCN_ALIGN_1BYTES)
                .Case( "IMAGE_SCN_ALIGN_2BYTES"
                     , COFF::IMAGE_SCN_ALIGN_2BYTES)
                .Case( "IMAGE_SCN_ALIGN_4BYTES"
                     , COFF::IMAGE_SCN_ALIGN_4BYTES)
                .Case( "IMAGE_SCN_ALIGN_8BYTES"
                     , COFF::IMAGE_SCN_ALIGN_8BYTES)
                .Case( "IMAGE_SCN_ALIGN_16BYTES"
                     , COFF::IMAGE_SCN_ALIGN_16BYTES)
                .Case( "IMAGE_SCN_ALIGN_32BYTES"
                     , COFF::IMAGE_SCN_ALIGN_32BYTES)
                .Case( "IMAGE_SCN_ALIGN_64BYTES"
                     , COFF::IMAGE_SCN_ALIGN_64BYTES)
                .Case( "IMAGE_SCN_ALIGN_128BYTES"
                     , COFF::IMAGE_SCN_ALIGN_128BYTES)
                .Case( "IMAGE_SCN_ALIGN_256BYTES"
                     , COFF::IMAGE_SCN_ALIGN_256BYTES)
                .Case( "IMAGE_SCN_ALIGN_512BYTES"
                     , COFF::IMAGE_SCN_ALIGN_512BYTES)
                .Case( "IMAGE_SCN_ALIGN_1024BYTES"
                     , COFF::IMAGE_SCN_ALIGN_1024BYTES)
                .Case( "IMAGE_SCN_ALIGN_2048BYTES"
                     , COFF::IMAGE_SCN_ALIGN_2048BYTES)
                .Case( "IMAGE_SCN_ALIGN_4096BYTES"
                     , COFF::IMAGE_SCN_ALIGN_4096BYTES)
                .Case( "IMAGE_SCN_ALIGN_8192BYTES"
                     , COFF::IMAGE_SCN_ALIGN_8192BYTES)
                .Case( "IMAGE_SCN_LNK_NRELOC_OVFL"
                     , COFF::IMAGE_SCN_LNK_NRELOC_OVFL)
                .Case( "IMAGE_SCN_MEM_DISCARDABLE"
                     , COFF::IMAGE_SCN_MEM_DISCARDABLE)
                .Case( "IMAGE_SCN_MEM_NOT_CACHED"
                     , COFF::IMAGE_SCN_MEM_NOT_CACHED)
                .Case( "IMAGE_SCN_MEM_NOT_PAGED"
                     , COFF::IMAGE_SCN_MEM_NOT_PAGED)
                .Case( "IMAGE_SCN_MEM_SHARED"
                     , COFF::IMAGE_SCN_MEM_SHARED)
                .Case( "IMAGE_SCN_MEM_EXECUTE"
                     , COFF::IMAGE_SCN_MEM_EXECUTE)
                .Case( "IMAGE_SCN_MEM_READ"
                     , COFF::IMAGE_SCN_MEM_READ)
                .Case( "IMAGE_SCN_MEM_WRITE"
                     , COFF::IMAGE_SCN_MEM_WRITE)
                .Default(COFF::SC_Invalid);
              if (Characteristic == COFF::SC_Invalid) {
                YS.printError(CharValue, "Invalid value for Characteristic");
                return false;
              }
              Sec.Header.Characteristics |= Characteristic;
            }
          }
        } else if (KeyValue == "SectionData") {
          yaml::ScalarNode *Value = dyn_cast<yaml::ScalarNode>(si->getValue());
          SmallString<32> Storage;
          StringRef Data = Value->getValue(Storage);
          if (!hexStringToByteArray(Data, Sec.Data)) {
            YS.printError(Value, "SectionData must be a collection of pairs of"
                                 "hex bytes");
            return false;
          }
        } else
          si->skip();
      }
      Sections.push_back(Sec);
    }
    return true;
  }

  bool parseSymbols(yaml::Node *SymbolsN) {
    yaml::SequenceNode *SN = dyn_cast<yaml::SequenceNode>(SymbolsN);
    if (!SN) {
      YS.printError(SymbolsN, "Symbols must be a sequence");
      return false;
    }
    for (yaml::SequenceNode::iterator i = SN->begin(), e = SN->end();
                                      i != e; ++i) {
      Symbol Sym;
      std::memset(&Sym.Header, 0, sizeof(Sym.Header));
      yaml::MappingNode *SymMap = dyn_cast<yaml::MappingNode>(&*i);
      if (!SymMap) {
        YS.printError(&*i, "Symbol must be a map");
        return false;
      }
      for (yaml::MappingNode::iterator si = SymMap->begin(), se = SymMap->end();
                                       si != se; ++si) {
        yaml::ScalarNode *Key = dyn_cast<yaml::ScalarNode>(si->getKey());
        if (!Key) {
          YS.printError(si->getKey(), "Keys must be scalar values");
          return false;
        }
        SmallString<32> Storage;
        StringRef KeyValue = Key->getValue(Storage);

        yaml::ScalarNode *Value = dyn_cast<yaml::ScalarNode>(si->getValue());
        if (!Value) {
          YS.printError(si->getValue(), "Must be a scalar value");
          return false;
        }
        if (KeyValue == "Name") {
          // If the name is less than 8 bytes, store it in place, otherwise
          // store it in the string table.
          StringRef Name = Value->getValue(Storage);
          std::fill_n(Sym.Header.Name, unsigned(COFF::NameSize), 0);
          if (Name.size() <= COFF::NameSize) {
            std::copy(Name.begin(), Name.end(), Sym.Header.Name);
          } else {
            // Add string to the string table and format the index for output.
            unsigned Index = getStringIndex(Name);
            *reinterpret_cast<support::aligned_ulittle32_t*>(
              Sym.Header.Name + 4) = Index;
          }
        } else if (KeyValue == "Value") {
          if (!getAs(Value, Sym.Header.Value)) {
            YS.printError(Value, "Invalid value for Value");
            return false;
          }
        } else if (KeyValue == "SimpleType") {
          Sym.Header.Type |= StringSwitch<COFF::SymbolBaseType>(
            Value->getValue(Storage))
            .Case("IMAGE_SYM_TYPE_NULL", COFF::IMAGE_SYM_TYPE_NULL)
            .Case("IMAGE_SYM_TYPE_VOID", COFF::IMAGE_SYM_TYPE_VOID)
            .Case("IMAGE_SYM_TYPE_CHAR", COFF::IMAGE_SYM_TYPE_CHAR)
            .Case("IMAGE_SYM_TYPE_SHORT", COFF::IMAGE_SYM_TYPE_SHORT)
            .Case("IMAGE_SYM_TYPE_INT", COFF::IMAGE_SYM_TYPE_INT)
            .Case("IMAGE_SYM_TYPE_LONG", COFF::IMAGE_SYM_TYPE_LONG)
            .Case("IMAGE_SYM_TYPE_FLOAT", COFF::IMAGE_SYM_TYPE_FLOAT)
            .Case("IMAGE_SYM_TYPE_DOUBLE", COFF::IMAGE_SYM_TYPE_DOUBLE)
            .Case("IMAGE_SYM_TYPE_STRUCT", COFF::IMAGE_SYM_TYPE_STRUCT)
            .Case("IMAGE_SYM_TYPE_UNION", COFF::IMAGE_SYM_TYPE_UNION)
            .Case("IMAGE_SYM_TYPE_ENUM", COFF::IMAGE_SYM_TYPE_ENUM)
            .Case("IMAGE_SYM_TYPE_MOE", COFF::IMAGE_SYM_TYPE_MOE)
            .Case("IMAGE_SYM_TYPE_BYTE", COFF::IMAGE_SYM_TYPE_BYTE)
            .Case("IMAGE_SYM_TYPE_WORD", COFF::IMAGE_SYM_TYPE_WORD)
            .Case("IMAGE_SYM_TYPE_UINT", COFF::IMAGE_SYM_TYPE_UINT)
            .Case("IMAGE_SYM_TYPE_DWORD", COFF::IMAGE_SYM_TYPE_DWORD)
            .Default(COFF::IMAGE_SYM_TYPE_NULL);
        } else if (KeyValue == "ComplexType") {
          Sym.Header.Type |= StringSwitch<COFF::SymbolComplexType>(
            Value->getValue(Storage))
            .Case("IMAGE_SYM_DTYPE_NULL", COFF::IMAGE_SYM_DTYPE_NULL)
            .Case("IMAGE_SYM_DTYPE_POINTER", COFF::IMAGE_SYM_DTYPE_POINTER)
            .Case("IMAGE_SYM_DTYPE_FUNCTION", COFF::IMAGE_SYM_DTYPE_FUNCTION)
            .Case("IMAGE_SYM_DTYPE_ARRAY", COFF::IMAGE_SYM_DTYPE_ARRAY)
            .Default(COFF::IMAGE_SYM_DTYPE_NULL)
            << COFF::SCT_COMPLEX_TYPE_SHIFT;
        } else if (KeyValue == "StorageClass") {
          Sym.Header.StorageClass = StringSwitch<COFF::SymbolStorageClass>(
            Value->getValue(Storage))
            .Case( "IMAGE_SYM_CLASS_END_OF_FUNCTION"
                 , COFF::IMAGE_SYM_CLASS_END_OF_FUNCTION)
            .Case( "IMAGE_SYM_CLASS_NULL"
                 , COFF::IMAGE_SYM_CLASS_NULL)
            .Case( "IMAGE_SYM_CLASS_AUTOMATIC"
                 , COFF::IMAGE_SYM_CLASS_AUTOMATIC)
            .Case( "IMAGE_SYM_CLASS_EXTERNAL"
                 , COFF::IMAGE_SYM_CLASS_EXTERNAL)
            .Case( "IMAGE_SYM_CLASS_STATIC"
                 , COFF::IMAGE_SYM_CLASS_STATIC)
            .Case( "IMAGE_SYM_CLASS_REGISTER"
                 , COFF::IMAGE_SYM_CLASS_REGISTER)
            .Case( "IMAGE_SYM_CLASS_EXTERNAL_DEF"
                 , COFF::IMAGE_SYM_CLASS_EXTERNAL_DEF)
            .Case( "IMAGE_SYM_CLASS_LABEL"
                 , COFF::IMAGE_SYM_CLASS_LABEL)
            .Case( "IMAGE_SYM_CLASS_UNDEFINED_LABEL"
                 , COFF::IMAGE_SYM_CLASS_UNDEFINED_LABEL)
            .Case( "IMAGE_SYM_CLASS_MEMBER_OF_STRUCT"
                 , COFF::IMAGE_SYM_CLASS_MEMBER_OF_STRUCT)
            .Case( "IMAGE_SYM_CLASS_ARGUMENT"
                 , COFF::IMAGE_SYM_CLASS_ARGUMENT)
            .Case( "IMAGE_SYM_CLASS_STRUCT_TAG"
                 , COFF::IMAGE_SYM_CLASS_STRUCT_TAG)
            .Case( "IMAGE_SYM_CLASS_MEMBER_OF_UNION"
                 , COFF::IMAGE_SYM_CLASS_MEMBER_OF_UNION)
            .Case( "IMAGE_SYM_CLASS_UNION_TAG"
                 , COFF::IMAGE_SYM_CLASS_UNION_TAG)
            .Case( "IMAGE_SYM_CLASS_TYPE_DEFINITION"
                 , COFF::IMAGE_SYM_CLASS_TYPE_DEFINITION)
            .Case( "IMAGE_SYM_CLASS_UNDEFINED_STATIC"
                 , COFF::IMAGE_SYM_CLASS_UNDEFINED_STATIC)
            .Case( "IMAGE_SYM_CLASS_ENUM_TAG"
                 , COFF::IMAGE_SYM_CLASS_ENUM_TAG)
            .Case( "IMAGE_SYM_CLASS_MEMBER_OF_ENUM"
                 , COFF::IMAGE_SYM_CLASS_MEMBER_OF_ENUM)
            .Case( "IMAGE_SYM_CLASS_REGISTER_PARAM"
                 , COFF::IMAGE_SYM_CLASS_REGISTER_PARAM)
            .Case( "IMAGE_SYM_CLASS_BIT_FIELD"
                 , COFF::IMAGE_SYM_CLASS_BIT_FIELD)
            .Case( "IMAGE_SYM_CLASS_BLOCK"
                 , COFF::IMAGE_SYM_CLASS_BLOCK)
            .Case( "IMAGE_SYM_CLASS_FUNCTION"
                 , COFF::IMAGE_SYM_CLASS_FUNCTION)
            .Case( "IMAGE_SYM_CLASS_END_OF_STRUCT"
                 , COFF::IMAGE_SYM_CLASS_END_OF_STRUCT)
            .Case( "IMAGE_SYM_CLASS_FILE"
                 , COFF::IMAGE_SYM_CLASS_FILE)
            .Case( "IMAGE_SYM_CLASS_SECTION"
                 , COFF::IMAGE_SYM_CLASS_SECTION)
            .Case( "IMAGE_SYM_CLASS_WEAK_EXTERNAL"
                 , COFF::IMAGE_SYM_CLASS_WEAK_EXTERNAL)
            .Case( "IMAGE_SYM_CLASS_CLR_TOKEN"
                 , COFF::IMAGE_SYM_CLASS_CLR_TOKEN)
            .Default(COFF::SSC_Invalid);
          if (Sym.Header.StorageClass == COFF::SSC_Invalid) {
            YS.printError(Value, "Invalid value for StorageClass");
            return false;
          }
        } else if (KeyValue == "SectionNumber") {
          if (!getAs(Value, Sym.Header.SectionNumber)) {
              YS.printError(Value, "Invalid value for SectionNumber");
              return false;
          }
        } else if (KeyValue == "AuxillaryData") {
          StringRef Data = Value->getValue(Storage);
          if (!hexStringToByteArray(Data, Sym.AuxSymbols)) {
            YS.printError(Value, "AuxillaryData must be a collection of pairs"
                                 "of hex bytes");
            return false;
          }
        } else
          si->skip();
      }
      Symbols.push_back(Sym);
    }
    return true;
  }

  bool parse() {
    yaml::Document &D = *YS.begin();
    yaml::MappingNode *Root = dyn_cast<yaml::MappingNode>(D.getRoot());
    if (!Root) {
      YS.printError(D.getRoot(), "Root node must be a map");
      return false;
    }
    for (yaml::MappingNode::iterator i = Root->begin(), e = Root->end();
                                     i != e; ++i) {
      yaml::ScalarNode *Key = dyn_cast<yaml::ScalarNode>(i->getKey());
      if (!Key) {
        YS.printError(i->getKey(), "Keys must be scalar values");
        return false;
      }
      SmallString<32> Storage;
      StringRef KeyValue = Key->getValue(Storage);
      if (KeyValue == "header") {
        if (!parseHeader(i->getValue()))
          return false;
      } else if (KeyValue == "sections") {
        if (!parseSections(i->getValue()))
          return false;
      } else if (KeyValue == "symbols") {
        if (!parseSymbols(i->getValue()))
          return false;
      }
    }
    return !YS.failed();
  }

  unsigned getStringIndex(StringRef Str) {
    StringMap<unsigned>::iterator i = StringTableMap.find(Str);
    if (i == StringTableMap.end()) {
      unsigned Index = StringTable.size();
      StringTable.append(Str.begin(), Str.end());
      StringTable.push_back(0);
      StringTableMap[Str] = Index;
      return Index;
    }
    return i->second;
  }

  yaml::Stream &YS;
  COFF::header Header;

  struct Section {
    COFF::section Header;
    std::vector<uint8_t> Data;
    std::vector<COFF::relocation> Relocations;
  };

  struct Symbol {
    COFF::symbol Header;
    std::vector<uint8_t> AuxSymbols;
  };

  std::vector<Section> Sections;
  std::vector<Symbol> Symbols;
  StringMap<unsigned> StringTableMap;
  std::string StringTable;
};

// Take a CP and assign addresses and sizes to everything. Returns false if the
// layout is not valid to do.
static bool layoutCOFF(COFFParser &CP) {
  uint32_t SectionTableStart = 0;
  uint32_t SectionTableSize  = 0;

  // The section table starts immediately after the header, including the
  // optional header.
  SectionTableStart = sizeof(COFF::header) + CP.Header.SizeOfOptionalHeader;
  SectionTableSize = sizeof(COFF::section) * CP.Sections.size();

  uint32_t CurrentSectionDataOffset = SectionTableStart + SectionTableSize;

  // Assign each section data address consecutively.
  for (std::vector<COFFParser::Section>::iterator i = CP.Sections.begin(),
                                                  e = CP.Sections.end();
                                                  i != e; ++i) {
    if (!i->Data.empty()) {
      i->Header.SizeOfRawData = i->Data.size();
      i->Header.PointerToRawData = CurrentSectionDataOffset;
      CurrentSectionDataOffset += i->Header.SizeOfRawData;
      // TODO: Handle alignment.
    } else {
      i->Header.SizeOfRawData = 0;
      i->Header.PointerToRawData = 0;
    }
  }

  uint32_t SymbolTableStart = CurrentSectionDataOffset;

  // Calculate number of symbols.
  uint32_t NumberOfSymbols = 0;
  for (std::vector<COFFParser::Symbol>::iterator i = CP.Symbols.begin(),
                                                 e = CP.Symbols.end();
                                                 i != e; ++i) {
    if (i->AuxSymbols.size() % COFF::SymbolSize != 0) {
      errs() << "AuxillaryData size not a multiple of symbol size!\n";
      return false;
    }
    i->Header.NumberOfAuxSymbols = i->AuxSymbols.size() / COFF::SymbolSize;
    NumberOfSymbols += 1 + i->Header.NumberOfAuxSymbols;
  }

  // Store all the allocated start addresses in the header.
  CP.Header.NumberOfSections = CP.Sections.size();
  CP.Header.NumberOfSymbols = NumberOfSymbols;
  CP.Header.PointerToSymbolTable = SymbolTableStart;

  *reinterpret_cast<support::ulittle32_t *>(&CP.StringTable[0])
    = CP.StringTable.size();

  return true;
}

template <typename value_type>
struct binary_le_impl {
  value_type Value;
  binary_le_impl(value_type V) : Value(V) {}
};

template <typename value_type>
raw_ostream &operator <<( raw_ostream &OS
                        , const binary_le_impl<value_type> &BLE) {
  char Buffer[sizeof(BLE.Value)];
  support::endian::write_le<value_type, support::unaligned>(Buffer, BLE.Value);
  OS.write(Buffer, sizeof(BLE.Value));
  return OS;
}

template <typename value_type>
binary_le_impl<value_type> binary_le(value_type V) {
  return binary_le_impl<value_type>(V);
}

void writeCOFF(COFFParser &CP, raw_ostream &OS) {
  OS << binary_le(CP.Header.Machine)
     << binary_le(CP.Header.NumberOfSections)
     << binary_le(CP.Header.TimeDateStamp)
     << binary_le(CP.Header.PointerToSymbolTable)
     << binary_le(CP.Header.NumberOfSymbols)
     << binary_le(CP.Header.SizeOfOptionalHeader)
     << binary_le(CP.Header.Characteristics);

  // Output section table.
  for (std::vector<COFFParser::Section>::const_iterator i = CP.Sections.begin(),
                                                        e = CP.Sections.end();
                                                        i != e; ++i) {
    OS.write(i->Header.Name, COFF::NameSize);
    OS << binary_le(i->Header.VirtualSize)
       << binary_le(i->Header.VirtualAddress)
       << binary_le(i->Header.SizeOfRawData)
       << binary_le(i->Header.PointerToRawData)
       << binary_le(i->Header.PointerToRelocations)
       << binary_le(i->Header.PointerToLineNumbers)
       << binary_le(i->Header.NumberOfRelocations)
       << binary_le(i->Header.NumberOfLineNumbers)
       << binary_le(i->Header.Characteristics);
  }

  // Output section data.
  for (std::vector<COFFParser::Section>::const_iterator i = CP.Sections.begin(),
                                                        e = CP.Sections.end();
                                                        i != e; ++i) {
    if (!i->Data.empty())
      OS.write(reinterpret_cast<const char*>(&i->Data[0]), i->Data.size());
  }

  // Output symbol table.

  for (std::vector<COFFParser::Symbol>::const_iterator i = CP.Symbols.begin(),
                                                       e = CP.Symbols.end();
                                                       i != e; ++i) {
    OS.write(i->Header.Name, COFF::NameSize);
    OS << binary_le(i->Header.Value)
       << binary_le(i->Header.SectionNumber)
       << binary_le(i->Header.Type)
       << binary_le(i->Header.StorageClass)
       << binary_le(i->Header.NumberOfAuxSymbols);
    if (!i->AuxSymbols.empty())
      OS.write( reinterpret_cast<const char*>(&i->AuxSymbols[0])
              , i->AuxSymbols.size());
  }

  // Output string table.
  OS.write(&CP.StringTable[0], CP.StringTable.size());
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  OwningPtr<MemoryBuffer> Buf;
  if (MemoryBuffer::getFileOrSTDIN(Input, Buf))
    return 1;

  SourceMgr SM;
  yaml::Stream S(Buf->getBuffer(), SM);
  COFFParser CP(S);
  if (!CP.parse()) {
    errs() << "yaml2obj: Failed to parse YAML file!\n";
    return 1;
  }
  if (!layoutCOFF(CP)) {
    errs() << "yaml2obj: Failed to layout COFF file!\n";
    return 1;
  }
  writeCOFF(CP, outs());
}
