//===- DWARFFormValue.cpp -------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SyntaxHighlighting.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFRelocMap.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cinttypes>
#include <cstdint>
#include <limits>

using namespace llvm;
using namespace dwarf;
using namespace syntax;

static const DWARFFormValue::FormClass DWARF4FormClasses[] = {
  DWARFFormValue::FC_Unknown,       // 0x0
  DWARFFormValue::FC_Address,       // 0x01 DW_FORM_addr
  DWARFFormValue::FC_Unknown,       // 0x02 unused
  DWARFFormValue::FC_Block,         // 0x03 DW_FORM_block2
  DWARFFormValue::FC_Block,         // 0x04 DW_FORM_block4
  DWARFFormValue::FC_Constant,      // 0x05 DW_FORM_data2
  // --- These can be FC_SectionOffset in DWARF3 and below:
  DWARFFormValue::FC_Constant,      // 0x06 DW_FORM_data4
  DWARFFormValue::FC_Constant,      // 0x07 DW_FORM_data8
  // ---
  DWARFFormValue::FC_String,        // 0x08 DW_FORM_string
  DWARFFormValue::FC_Block,         // 0x09 DW_FORM_block
  DWARFFormValue::FC_Block,         // 0x0a DW_FORM_block1
  DWARFFormValue::FC_Constant,      // 0x0b DW_FORM_data1
  DWARFFormValue::FC_Flag,          // 0x0c DW_FORM_flag
  DWARFFormValue::FC_Constant,      // 0x0d DW_FORM_sdata
  DWARFFormValue::FC_String,        // 0x0e DW_FORM_strp
  DWARFFormValue::FC_Constant,      // 0x0f DW_FORM_udata
  DWARFFormValue::FC_Reference,     // 0x10 DW_FORM_ref_addr
  DWARFFormValue::FC_Reference,     // 0x11 DW_FORM_ref1
  DWARFFormValue::FC_Reference,     // 0x12 DW_FORM_ref2
  DWARFFormValue::FC_Reference,     // 0x13 DW_FORM_ref4
  DWARFFormValue::FC_Reference,     // 0x14 DW_FORM_ref8
  DWARFFormValue::FC_Reference,     // 0x15 DW_FORM_ref_udata
  DWARFFormValue::FC_Indirect,      // 0x16 DW_FORM_indirect
  DWARFFormValue::FC_SectionOffset, // 0x17 DW_FORM_sec_offset
  DWARFFormValue::FC_Exprloc,       // 0x18 DW_FORM_exprloc
  DWARFFormValue::FC_Flag,          // 0x19 DW_FORM_flag_present
};

namespace {

/// A helper class that can be used in DWARFFormValue.cpp functions that need
/// to know the byte size of DW_FORM values that vary in size depending on the
/// DWARF version, address byte size, or DWARF32 or DWARF64.
class FormSizeHelper {
  uint16_t Version;
  uint8_t AddrSize;
  llvm::dwarf::DwarfFormat Format;

public:
  FormSizeHelper(uint16_t V, uint8_t A, llvm::dwarf::DwarfFormat F)
      : Version(V), AddrSize(A), Format(F) {}

  uint8_t getAddressByteSize() const { return AddrSize; }

  uint8_t getRefAddrByteSize() const {
    if (Version == 2)
      return AddrSize;
    return getDwarfOffsetByteSize();
  }

  uint8_t getDwarfOffsetByteSize() const {
    switch (Format) {
      case dwarf::DwarfFormat::DWARF32:
        return 4;
      case dwarf::DwarfFormat::DWARF64:
        return 8;
    }
    llvm_unreachable("Invalid Format value");
  }
};

} // end anonymous namespace

template <class T>
static Optional<uint8_t> getFixedByteSize(dwarf::Form Form, const T *U) {
  switch (Form) {
    case DW_FORM_addr:
      if (U)
        return U->getAddressByteSize();
      return None;

    case DW_FORM_block:          // ULEB128 length L followed by L bytes.
    case DW_FORM_block1:         // 1 byte length L followed by L bytes.
    case DW_FORM_block2:         // 2 byte length L followed by L bytes.
    case DW_FORM_block4:         // 4 byte length L followed by L bytes.
    case DW_FORM_string:         // C-string with null terminator.
    case DW_FORM_sdata:          // SLEB128.
    case DW_FORM_udata:          // ULEB128.
    case DW_FORM_ref_udata:      // ULEB128.
    case DW_FORM_indirect:       // ULEB128.
    case DW_FORM_exprloc:        // ULEB128 length L followed by L bytes.
    case DW_FORM_strx:           // ULEB128.
    case DW_FORM_addrx:          // ULEB128.
    case DW_FORM_loclistx:       // ULEB128.
    case DW_FORM_rnglistx:       // ULEB128.
    case DW_FORM_GNU_addr_index: // ULEB128.
    case DW_FORM_GNU_str_index:  // ULEB128.
      return None;

    case DW_FORM_ref_addr:
      if (U)
        return U->getRefAddrByteSize();
      return None;

    case DW_FORM_flag:
    case DW_FORM_data1:
    case DW_FORM_ref1:
    case DW_FORM_strx1:
    case DW_FORM_addrx1:
      return 1;

    case DW_FORM_data2:
    case DW_FORM_ref2:
    case DW_FORM_strx2:
    case DW_FORM_addrx2:
      return 2;

    case DW_FORM_data4:
    case DW_FORM_ref4:
    case DW_FORM_ref_sup4:
    case DW_FORM_strx4:
    case DW_FORM_addrx4:
      return 4;

    case DW_FORM_strp:
    case DW_FORM_GNU_ref_alt:
    case DW_FORM_GNU_strp_alt:
    case DW_FORM_line_strp:
    case DW_FORM_sec_offset:
    case DW_FORM_strp_sup:
      if (U)
        return U->getDwarfOffsetByteSize();
      return None;

    case DW_FORM_data8:
    case DW_FORM_ref8:
    case DW_FORM_ref_sig8:
    case DW_FORM_ref_sup8:
      return 8;

    case DW_FORM_flag_present:
      return 0;

    case DW_FORM_data16:
      return 16;

    case DW_FORM_implicit_const:
      // The implicit value is stored in the abbreviation as a SLEB128, and
      // there no data in debug info.
      return 0;

    default:
      llvm_unreachable("Handle this form in this switch statement");
  }
  return None;
}

template <class T>
static bool skipFormValue(dwarf::Form Form, const DataExtractor &DebugInfoData,
                          uint32_t *OffsetPtr, const T *U) {
  bool Indirect = false;
  do {
    switch (Form) {
        // Blocks of inlined data that have a length field and the data bytes
        // inlined in the .debug_info.
      case DW_FORM_exprloc:
      case DW_FORM_block: {
        uint64_t size = DebugInfoData.getULEB128(OffsetPtr);
        *OffsetPtr += size;
        return true;
      }
      case DW_FORM_block1: {
        uint8_t size = DebugInfoData.getU8(OffsetPtr);
        *OffsetPtr += size;
        return true;
      }
      case DW_FORM_block2: {
        uint16_t size = DebugInfoData.getU16(OffsetPtr);
        *OffsetPtr += size;
        return true;
      }
      case DW_FORM_block4: {
        uint32_t size = DebugInfoData.getU32(OffsetPtr);
        *OffsetPtr += size;
        return true;
      }

        // Inlined NULL terminated C-strings.
      case DW_FORM_string:
        DebugInfoData.getCStr(OffsetPtr);
        return true;

      case DW_FORM_addr:
      case DW_FORM_ref_addr:
      case DW_FORM_flag_present:
      case DW_FORM_data1:
      case DW_FORM_data2:
      case DW_FORM_data4:
      case DW_FORM_data8:
      case DW_FORM_flag:
      case DW_FORM_ref1:
      case DW_FORM_ref2:
      case DW_FORM_ref4:
      case DW_FORM_ref8:
      case DW_FORM_ref_sig8:
      case DW_FORM_ref_sup4:
      case DW_FORM_ref_sup8:
      case DW_FORM_strx1:
      case DW_FORM_strx2:
      case DW_FORM_strx4:
      case DW_FORM_addrx1:
      case DW_FORM_addrx2:
      case DW_FORM_addrx4:
      case DW_FORM_sec_offset:
      case DW_FORM_strp:
      case DW_FORM_strp_sup:
      case DW_FORM_line_strp:
      case DW_FORM_GNU_ref_alt:
      case DW_FORM_GNU_strp_alt:
        if (Optional<uint8_t> FixedSize = ::getFixedByteSize(Form, U)) {
          *OffsetPtr += *FixedSize;
          return true;
        }
        return false;

        // signed or unsigned LEB 128 values.
      case DW_FORM_sdata:
        DebugInfoData.getSLEB128(OffsetPtr);
        return true;

      case DW_FORM_udata:
      case DW_FORM_ref_udata:
      case DW_FORM_strx:
      case DW_FORM_addrx:
      case DW_FORM_loclistx:
      case DW_FORM_rnglistx:
      case DW_FORM_GNU_addr_index:
      case DW_FORM_GNU_str_index:
        DebugInfoData.getULEB128(OffsetPtr);
        return true;
        
      case DW_FORM_indirect:
        Indirect = true;
        Form = static_cast<dwarf::Form>(DebugInfoData.getULEB128(OffsetPtr));
        break;
        
      default:
        return false;
    }
  } while (Indirect);
  return true;
}

Optional<uint8_t> DWARFFormValue::getFixedByteSize(dwarf::Form Form,
                                                   const DWARFUnit *U) {
  return ::getFixedByteSize(Form, U);
}

Optional<uint8_t>
DWARFFormValue::getFixedByteSize(dwarf::Form Form, uint16_t Version,
                                 uint8_t AddrSize,
                                 llvm::dwarf::DwarfFormat Format) {
  FormSizeHelper FSH(Version, AddrSize, Format);
  return ::getFixedByteSize(Form, &FSH);
}

bool DWARFFormValue::isFormClass(DWARFFormValue::FormClass FC) const {
  // First, check DWARF4 form classes.
  if (Form < makeArrayRef(DWARF4FormClasses).size() &&
      DWARF4FormClasses[Form] == FC)
    return true;
  // Check more forms from DWARF4 and DWARF5 proposals.
  switch (Form) {
  case DW_FORM_ref_sig8:
  case DW_FORM_GNU_ref_alt:
    return (FC == FC_Reference);
  case DW_FORM_GNU_addr_index:
    return (FC == FC_Address);
  case DW_FORM_GNU_str_index:
  case DW_FORM_GNU_strp_alt:
    return (FC == FC_String);
  case DW_FORM_implicit_const:
    return (FC == FC_Constant);
  default:
    break;
  }
  // In DWARF3 DW_FORM_data4 and DW_FORM_data8 served also as a section offset.
  // Don't check for DWARF version here, as some producers may still do this
  // by mistake.
  return (Form == DW_FORM_data4 || Form == DW_FORM_data8) &&
         FC == FC_SectionOffset;
}

bool DWARFFormValue::extractValue(const DataExtractor &data, 
                                  uint32_t *offset_ptr,
                                  const DWARFUnit *cu) {
  U = cu;
  bool indirect = false;
  bool is_block = false;
  Value.data = nullptr;
  // Read the value for the form into value and follow and DW_FORM_indirect
  // instances we run into
  do {
    indirect = false;
    switch (Form) {
    case DW_FORM_addr:
    case DW_FORM_ref_addr: {
      if (!U)
        return false;
      uint16_t AddrSize =
          (Form == DW_FORM_addr)
              ? U->getAddressByteSize()
              : U->getRefAddrByteSize();
      Value.uval =
          getRelocatedValue(data, AddrSize, offset_ptr, U->getRelocMap());
      break;
    }
    case DW_FORM_exprloc:
    case DW_FORM_block:
      Value.uval = data.getULEB128(offset_ptr);
      is_block = true;
      break;
    case DW_FORM_block1:
      Value.uval = data.getU8(offset_ptr);
      is_block = true;
      break;
    case DW_FORM_block2:
      Value.uval = data.getU16(offset_ptr);
      is_block = true;
      break;
    case DW_FORM_block4:
      Value.uval = data.getU32(offset_ptr);
      is_block = true;
      break;
    case DW_FORM_data1:
    case DW_FORM_ref1:
    case DW_FORM_flag:
    case DW_FORM_strx1:
    case DW_FORM_addrx1:
      Value.uval = data.getU8(offset_ptr);
      break;
    case DW_FORM_data2:
    case DW_FORM_ref2:
    case DW_FORM_strx2:
    case DW_FORM_addrx2:
      Value.uval = data.getU16(offset_ptr);
      break;
    case DW_FORM_data4:
    case DW_FORM_ref4:
    case DW_FORM_ref_sup4:
    case DW_FORM_strx4:
    case DW_FORM_addrx4: {
      const RelocAddrMap* RelocMap = U ? U->getRelocMap() : nullptr;
      Value.uval = getRelocatedValue(data, 4, offset_ptr, RelocMap);
      break;
    }
    case DW_FORM_data8:
    case DW_FORM_ref8:
    case DW_FORM_ref_sup8:
      Value.uval = data.getU64(offset_ptr);
      break;
    case DW_FORM_sdata:
      Value.sval = data.getSLEB128(offset_ptr);
      break;
    case DW_FORM_udata:
    case DW_FORM_ref_udata:
      Value.uval = data.getULEB128(offset_ptr);
      break;
    case DW_FORM_string:
      Value.cstr = data.getCStr(offset_ptr);
      break;
    case DW_FORM_indirect:
      Form = static_cast<dwarf::Form>(data.getULEB128(offset_ptr));
      indirect = true;
      break;
    case DW_FORM_strp:
    case DW_FORM_sec_offset:
    case DW_FORM_GNU_ref_alt:
    case DW_FORM_GNU_strp_alt:
    case DW_FORM_line_strp:
    case DW_FORM_strp_sup: {
      if (!U)
        return false;
      Value.uval = getRelocatedValue(data, U->getDwarfOffsetByteSize(),
                                     offset_ptr, U->getRelocMap());
      break;
    }
    case DW_FORM_flag_present:
      Value.uval = 1;
      break;
    case DW_FORM_ref_sig8:
      Value.uval = data.getU64(offset_ptr);
      break;
    case DW_FORM_GNU_addr_index:
    case DW_FORM_GNU_str_index:
      Value.uval = data.getULEB128(offset_ptr);
      break;
    default:
      // DWARFFormValue::skipValue() will have caught this and caused all
      // DWARF DIEs to fail to be parsed, so this code is not be reachable.
      llvm_unreachable("unsupported form");
    }
  } while (indirect);

  if (is_block) {
    StringRef str = data.getData().substr(*offset_ptr, Value.uval);
    Value.data = nullptr;
    if (!str.empty()) {
      Value.data = reinterpret_cast<const uint8_t *>(str.data());
      *offset_ptr += Value.uval;
    }
  }

  return true;
}

bool DWARFFormValue::skipValue(DataExtractor DebugInfoData,
                               uint32_t *offset_ptr, const DWARFUnit *U) const {
  return DWARFFormValue::skipValue(Form, DebugInfoData, offset_ptr, U);
}

bool DWARFFormValue::skipValue(dwarf::Form form, DataExtractor DebugInfoData,
                               uint32_t *offset_ptr, const DWARFUnit *U) {
  return skipFormValue(form, DebugInfoData, offset_ptr, U);
}

bool DWARFFormValue::skipValue(dwarf::Form form, DataExtractor DebugInfoData,
                               uint32_t *offset_ptr, uint16_t Version,
                               uint8_t AddrSize,
                               llvm::dwarf::DwarfFormat Format) {
  FormSizeHelper FSH(Version, AddrSize, Format);
  return skipFormValue(form, DebugInfoData, offset_ptr, &FSH);
}

void
DWARFFormValue::dump(raw_ostream &OS) const {
  uint64_t uvalue = Value.uval;
  bool cu_relative_offset = false;

  switch (Form) {
  case DW_FORM_addr:      OS << format("0x%016" PRIx64, uvalue); break;
  case DW_FORM_GNU_addr_index: {
    OS << format(" indexed (%8.8x) address = ", (uint32_t)uvalue);
    uint64_t Address;
    if (U == nullptr)
      OS << "<invalid dwarf unit>";
    else if (U->getAddrOffsetSectionItem(uvalue, Address))
      OS << format("0x%016" PRIx64, Address);
    else
      OS << "<no .debug_addr section>";
    break;
  }
  case DW_FORM_flag_present: OS << "true"; break;
  case DW_FORM_flag:
  case DW_FORM_data1:     OS << format("0x%02x", (uint8_t)uvalue); break;
  case DW_FORM_data2:     OS << format("0x%04x", (uint16_t)uvalue); break;
  case DW_FORM_data4:     OS << format("0x%08x", (uint32_t)uvalue); break;
  case DW_FORM_ref_sig8:
  case DW_FORM_data8:     OS << format("0x%016" PRIx64, uvalue); break;
  case DW_FORM_string:
    OS << '"';
    OS.write_escaped(Value.cstr);
    OS << '"';
    break;
  case DW_FORM_exprloc:
  case DW_FORM_block:
  case DW_FORM_block1:
  case DW_FORM_block2:
  case DW_FORM_block4:
    if (uvalue > 0) {
      switch (Form) {
      case DW_FORM_exprloc:
      case DW_FORM_block:  OS << format("<0x%" PRIx64 "> ", uvalue);     break;
      case DW_FORM_block1: OS << format("<0x%2.2x> ", (uint8_t)uvalue);  break;
      case DW_FORM_block2: OS << format("<0x%4.4x> ", (uint16_t)uvalue); break;
      case DW_FORM_block4: OS << format("<0x%8.8x> ", (uint32_t)uvalue); break;
      default: break;
      }

      const uint8_t* data_ptr = Value.data;
      if (data_ptr) {
        // uvalue contains size of block
        const uint8_t* end_data_ptr = data_ptr + uvalue;
        while (data_ptr < end_data_ptr) {
          OS << format("%2.2x ", *data_ptr);
          ++data_ptr;
        }
      }
      else
        OS << "NULL";
    }
    break;

  case DW_FORM_sdata:     OS << Value.sval; break;
  case DW_FORM_udata:     OS << Value.uval; break;
  case DW_FORM_strp:
    OS << format(" .debug_str[0x%8.8x] = ", (uint32_t)uvalue);
    dumpString(OS);
    break;
  case DW_FORM_GNU_str_index:
    OS << format(" indexed (%8.8x) string = ", (uint32_t)uvalue);
    dumpString(OS);
    break;
  case DW_FORM_GNU_strp_alt:
    OS << format("alt indirect string, offset: 0x%" PRIx64 "", uvalue);
    dumpString(OS);
    break;
  case DW_FORM_ref_addr:
    OS << format("0x%016" PRIx64, uvalue);
    break;
  case DW_FORM_ref1:
    cu_relative_offset = true;
    OS << format("cu + 0x%2.2x", (uint8_t)uvalue);
    break;
  case DW_FORM_ref2:
    cu_relative_offset = true;
    OS << format("cu + 0x%4.4x", (uint16_t)uvalue);
    break;
  case DW_FORM_ref4:
    cu_relative_offset = true;
    OS << format("cu + 0x%4.4x", (uint32_t)uvalue);
    break;
  case DW_FORM_ref8:
    cu_relative_offset = true;
    OS << format("cu + 0x%8.8" PRIx64, uvalue);
    break;
  case DW_FORM_ref_udata:
    cu_relative_offset = true;
    OS << format("cu + 0x%" PRIx64, uvalue);
    break;
  case DW_FORM_GNU_ref_alt:
    OS << format("<alt 0x%" PRIx64 ">", uvalue);
    break;

    // All DW_FORM_indirect attributes should be resolved prior to calling
    // this function
  case DW_FORM_indirect:
    OS << "DW_FORM_indirect";
    break;

    // Should be formatted to 64-bit for DWARF64.
  case DW_FORM_sec_offset:
    OS << format("0x%08x", (uint32_t)uvalue);
    break;

  default:
    OS << format("DW_FORM(0x%4.4x)", Form);
    break;
  }

  if (cu_relative_offset) {
    OS << " => {";
    WithColor(OS, syntax::Address).get()
      << format("0x%8.8" PRIx64, uvalue + (U ? U->getOffset() : 0));
    OS << "}";
  }
}

void DWARFFormValue::dumpString(raw_ostream &OS) const {
  Optional<const char *> DbgStr = getAsCString();
  if (DbgStr.hasValue()) {
    raw_ostream &COS = WithColor(OS, syntax::String);
    COS << '"';
    COS.write_escaped(DbgStr.getValue());
    COS << '"';
  }
}

Optional<const char *> DWARFFormValue::getAsCString() const {
  if (!isFormClass(FC_String))
    return None;
  if (Form == DW_FORM_string)
    return Value.cstr;
  // FIXME: Add support for DW_FORM_GNU_strp_alt
  if (Form == DW_FORM_GNU_strp_alt || U == nullptr)
    return None;
  uint32_t Offset = Value.uval;
  if (Form == DW_FORM_GNU_str_index) {
    uint32_t StrOffset;
    if (!U->getStringOffsetSectionItem(Offset, StrOffset))
      return None;
    Offset = StrOffset;
  }
  if (const char *Str = U->getStringExtractor().getCStr(&Offset)) {
    return Str;
  }
  return None;
}

Optional<uint64_t> DWARFFormValue::getAsAddress() const {
  if (!isFormClass(FC_Address))
    return None;
  if (Form == DW_FORM_GNU_addr_index) {
    uint32_t Index = Value.uval;
    uint64_t Result;
    if (!U || !U->getAddrOffsetSectionItem(Index, Result))
      return None;
    return Result;
  }
  return Value.uval;
}

Optional<uint64_t> DWARFFormValue::getAsReference() const {
  if (!isFormClass(FC_Reference))
    return None;
  switch (Form) {
  case DW_FORM_ref1:
  case DW_FORM_ref2:
  case DW_FORM_ref4:
  case DW_FORM_ref8:
  case DW_FORM_ref_udata:
    if (!U)
      return None;
    return Value.uval + U->getOffset();
  case DW_FORM_ref_addr:
  case DW_FORM_ref_sig8:
  case DW_FORM_GNU_ref_alt:
    return Value.uval;
  default:
    return None;
  }
}

Optional<uint64_t> DWARFFormValue::getAsSectionOffset() const {
  if (!isFormClass(FC_SectionOffset))
    return None;
  return Value.uval;
}

Optional<uint64_t> DWARFFormValue::getAsUnsignedConstant() const {
  if ((!isFormClass(FC_Constant) && !isFormClass(FC_Flag))
      || Form == DW_FORM_sdata)
    return None;
  return Value.uval;
}

Optional<int64_t> DWARFFormValue::getAsSignedConstant() const {
  if ((!isFormClass(FC_Constant) && !isFormClass(FC_Flag)) ||
      (Form == DW_FORM_udata && uint64_t(std::numeric_limits<int64_t>::max()) < Value.uval))
    return None;
  switch (Form) {
  case DW_FORM_data4:
    return int32_t(Value.uval);
  case DW_FORM_data2:
    return int16_t(Value.uval);
  case DW_FORM_data1:
    return int8_t(Value.uval);
  case DW_FORM_sdata:
  case DW_FORM_data8:
  default:
    return Value.sval;
  }
}

Optional<ArrayRef<uint8_t>> DWARFFormValue::getAsBlock() const {
  if (!isFormClass(FC_Block) && !isFormClass(FC_Exprloc))
    return None;
  return makeArrayRef(Value.data, Value.uval);
}

Optional<uint64_t> DWARFFormValue::getAsCStringOffset() const {
  if (!isFormClass(FC_String) && Form == DW_FORM_string)
    return None;
  return Value.uval;
}

Optional<uint64_t> DWARFFormValue::getAsReferenceUVal() const {
  if (!isFormClass(FC_Reference))
    return None;
  return Value.uval;
}
