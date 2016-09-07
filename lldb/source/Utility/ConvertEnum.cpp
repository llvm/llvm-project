//===-- ConvertEnum.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "lldb/Utility/ConvertEnum.h"

using namespace lldb;
using namespace lldb_private;

const char *lldb_private::GetVoteAsCString(Vote vote) {
  switch (vote) {
  case eVoteNo:
    return "no";
  case eVoteNoOpinion:
    return "no opinion";
  case eVoteYes:
    return "yes";
  }
  return "invalid";
}

const char *lldb_private::GetSectionTypeAsCString(lldb::SectionType sect_type) {
  switch (sect_type) {
  case eSectionTypeInvalid:
    return "invalid";
  case eSectionTypeCode:
    return "code";
  case eSectionTypeContainer:
    return "container";
  case eSectionTypeData:
    return "data";
  case eSectionTypeDataCString:
    return "data-cstr";
  case eSectionTypeDataCStringPointers:
    return "data-cstr-ptr";
  case eSectionTypeDataSymbolAddress:
    return "data-symbol-addr";
  case eSectionTypeData4:
    return "data-4-byte";
  case eSectionTypeData8:
    return "data-8-byte";
  case eSectionTypeData16:
    return "data-16-byte";
  case eSectionTypeDataPointers:
    return "data-ptrs";
  case eSectionTypeDebug:
    return "debug";
  case eSectionTypeZeroFill:
    return "zero-fill";
  case eSectionTypeDataObjCMessageRefs:
    return "objc-message-refs";
  case eSectionTypeDataObjCCFStrings:
    return "objc-cfstrings";
  case eSectionTypeDWARFDebugAbbrev:
    return "dwarf-abbrev";
  case eSectionTypeDWARFDebugAddr:
    return "dwarf-addr";
  case eSectionTypeDWARFDebugAranges:
    return "dwarf-aranges";
  case eSectionTypeDWARFDebugFrame:
    return "dwarf-frame";
  case eSectionTypeDWARFDebugInfo:
    return "dwarf-info";
  case eSectionTypeDWARFDebugLine:
    return "dwarf-line";
  case eSectionTypeDWARFDebugLoc:
    return "dwarf-loc";
  case eSectionTypeDWARFDebugMacInfo:
    return "dwarf-macinfo";
  case eSectionTypeDWARFDebugMacro:
    return "dwarf-macro";
  case eSectionTypeDWARFDebugPubNames:
    return "dwarf-pubnames";
  case eSectionTypeDWARFDebugPubTypes:
    return "dwarf-pubtypes";
  case eSectionTypeDWARFDebugRanges:
    return "dwarf-ranges";
  case eSectionTypeDWARFDebugStr:
    return "dwarf-str";
  case eSectionTypeDWARFDebugStrOffsets:
    return "dwarf-str-offsets";
  case eSectionTypeELFSymbolTable:
    return "elf-symbol-table";
  case eSectionTypeELFDynamicSymbols:
    return "elf-dynamic-symbols";
  case eSectionTypeELFRelocationEntries:
    return "elf-relocation-entries";
  case eSectionTypeELFDynamicLinkInfo:
    return "elf-dynamic-link-info";
  case eSectionTypeDWARFAppleNames:
    return "apple-names";
  case eSectionTypeDWARFAppleTypes:
    return "apple-types";
  case eSectionTypeDWARFAppleNamespaces:
    return "apple-namespaces";
  case eSectionTypeDWARFAppleObjC:
    return "apple-objc";
  case eSectionTypeEHFrame:
    return "eh-frame";
  case eSectionTypeARMexidx:
    return "ARM.exidx";
  case eSectionTypeARMextab:
    return "ARM.extab";
  case eSectionTypeCompactUnwind:
    return "compact-unwind";
  case eSectionTypeGoSymtab:
    return "go-symtab";
  case eSectionTypeAbsoluteAddress:
    return "absolute";
  case eSectionTypeOther:
    return "regular";
  }
  return "unknown";
}
