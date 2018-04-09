//===-- DWARFUnit.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFUnit.h"

#include "Plugins/Language/ObjC/ObjCLanguage.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/ObjectFile.h"

#include "DWARFCompileUnit.h"
#include "DWARFDebugInfo.h"
#include "LogChannelDWARF.h"
#include "SymbolFileDWARFDwo.h"

using namespace lldb;
using namespace lldb_private;
using namespace std;

extern int g_verbose;

DWARFUnit::DWARFUnit() {}

DWARFUnit::~DWARFUnit() {}

size_t DWARFUnit::ExtractDIEsIfNeeded(bool cu_die_only) {
  return Data().ExtractDIEsIfNeeded(cu_die_only);
}

DWARFDIE DWARFUnit::LookupAddress(const dw_addr_t address) {
  return Data().LookupAddress(address);
}

size_t DWARFUnit::AppendDIEsWithTag(const dw_tag_t tag,
				    DWARFDIECollection &dies,
				    uint32_t depth) const {
  return Data().AppendDIEsWithTag(tag, dies, depth);
}

bool DWARFUnit::Verify(Stream *s) const {
  return Data().Verify(s);
}

void DWARFUnit::Dump(Stream *s) const {
  Data().Dump(s);
}

lldb::user_id_t DWARFUnit::GetID() const {
  return Data().GetID();
}

uint32_t DWARFUnit::Size() const { return IsDWARF64() ? 23 : 11; }

dw_offset_t DWARFUnit::GetNextCompileUnitOffset() const {
  return m_offset + (IsDWARF64() ? 12 : 4) + GetLength();
}

size_t DWARFUnit::GetDebugInfoSize() const {
  return (IsDWARF64() ? 12 : 4) + GetLength() - Size();
}

uint32_t DWARFUnit::GetLength() const { return Data().m_length; }
uint16_t DWARFUnit::GetVersion() const { return Data().m_version; }

const DWARFAbbreviationDeclarationSet *DWARFUnit::GetAbbreviations() const {
  return Data().m_abbrevs;
}

dw_offset_t DWARFUnit::GetAbbrevOffset() const {
  return Data().GetAbbrevOffset();
}

uint8_t DWARFUnit::GetAddressByteSize() const { return Data().m_addr_size; }

dw_addr_t DWARFUnit::GetBaseAddress() const { return Data().m_base_addr; }

dw_addr_t DWARFUnit::GetAddrBase() const { return Data().m_addr_base; }

dw_addr_t DWARFUnit::GetRangesBase() const { return Data().m_ranges_base; }

void DWARFUnit::SetAddrBase(dw_addr_t addr_base,
                            dw_addr_t ranges_base,
                            dw_offset_t base_obj_offset) {
  Data().SetAddrBase(addr_base, ranges_base, base_obj_offset);
}

void DWARFUnit::ClearDIEs(bool keep_compile_unit_die) {
  Data().ClearDIEs(keep_compile_unit_die);
}

void DWARFUnit::BuildAddressRangeTable(SymbolFileDWARF *dwarf2Data,
                                       DWARFDebugAranges *debug_aranges) {
  Data().BuildAddressRangeTable(dwarf2Data, debug_aranges);
}

lldb::ByteOrder DWARFUnit::GetByteOrder() const {
  return Data().m_dwarf2Data->GetObjectFile()->GetByteOrder();
}

TypeSystem *DWARFUnit::GetTypeSystem() {
  return Data().GetTypeSystem();
}

DWARFFormValue::FixedFormSizes DWARFUnit::GetFixedFormSizes() {
  return DWARFFormValue::GetFixedFormSizesForAddressSize(GetAddressByteSize(),
                                                         IsDWARF64());
}

void DWARFUnit::SetBaseAddress(dw_addr_t base_addr) {
  Data().m_base_addr = base_addr;
}

DWARFDIE DWARFUnit::GetCompileUnitDIEOnly() {
  return Data().GetCompileUnitDIEOnly();
}

DWARFDIE DWARFUnit::DIE() {
  return Data().DIE();
}

bool DWARFUnit::HasDIEsParsed() const { return Data().m_die_array.size() > 1; }

//----------------------------------------------------------------------
// Compare function DWARFDebugAranges::Range structures
//----------------------------------------------------------------------
static bool CompareDIEOffset(const DWARFDebugInfoEntry &die,
                             const dw_offset_t die_offset) {
  return die.GetOffset() < die_offset;
}

//----------------------------------------------------------------------
// GetDIE()
//
// Get the DIE (Debug Information Entry) with the specified offset by
// first checking if the DIE is contained within this compile unit and
// grabbing the DIE from this compile unit. Otherwise we grab the DIE
// from the DWARF file.
//----------------------------------------------------------------------
DWARFDIE
DWARFUnit::GetDIE(dw_offset_t die_offset) {
  if (die_offset != DW_INVALID_OFFSET) {
    if (GetDwoSymbolFile())
      return GetDwoSymbolFile()->GetCompileUnit()->GetDIE(die_offset);

    if (ContainsDIEOffset(die_offset)) {
      ExtractDIEsIfNeeded(false);
      DWARFDebugInfoEntry::iterator end = Data().m_die_array.end();
      DWARFDebugInfoEntry::iterator pos = lower_bound(
          Data().m_die_array.begin(), end, die_offset, CompareDIEOffset);
      if (pos != end) {
        if (die_offset == (*pos).GetOffset())
          return DWARFDIE(this, &(*pos));
      }
    } else {
      // Don't specify the compile unit offset as we don't know it because the
      // DIE belongs to
      // a different compile unit in the same symbol file.
      return Data().m_dwarf2Data->DebugInfo()->GetDIEForDIEOffset(die_offset);
    }
  }
  return DWARFDIE(); // Not found
}

static uint8_t g_default_addr_size = 4;

uint8_t DWARFUnit::GetAddressByteSize(const DWARFUnit *cu) {
  if (cu)
    return cu->GetAddressByteSize();
  return DWARFCompileUnit::GetDefaultAddressSize();
}

bool DWARFUnit::IsDWARF64(const DWARFUnit *cu) {
  if (cu)
    return cu->IsDWARF64();
  return false;
}

uint8_t DWARFUnit::GetDefaultAddressSize() {
  return g_default_addr_size;
}

void DWARFUnit::SetDefaultAddressSize(uint8_t addr_size) {
  g_default_addr_size = addr_size;
}

void *DWARFUnit::GetUserData() const { return Data().m_user_data; }

void DWARFUnit::SetUserData(void *d) {
  Data().SetUserData(d);
}

bool DWARFUnit::Supports_DW_AT_APPLE_objc_complete_type() {
  if (GetProducer() == eProducerLLVMGCC)
    return false;
  return true;
}

bool DWARFUnit::DW_AT_decl_file_attributes_are_invalid() {
  // llvm-gcc makes completely invalid decl file attributes and won't ever
  // be fixed, so we need to know to ignore these.
  return GetProducer() == eProducerLLVMGCC;
}

bool DWARFUnit::Supports_unnamed_objc_bitfields() {
  if (GetProducer() == eProducerClang) {
    const uint32_t major_version = GetProducerVersionMajor();
    if (major_version > 425 ||
        (major_version == 425 && GetProducerVersionUpdate() >= 13))
      return true;
    else
      return false;
  }
  return true; // Assume all other compilers didn't have incorrect ObjC bitfield
               // info
}

SymbolFileDWARF *DWARFUnit::GetSymbolFileDWARF() const {
  return Data().m_dwarf2Data;
}

DWARFProducer DWARFUnit::GetProducer() {
  return Data().GetProducer();
}

uint32_t DWARFUnit::GetProducerVersionMajor() {
  return Data().GetProducerVersionMajor();
}

uint32_t DWARFUnit::GetProducerVersionMinor() {
  return Data().GetProducerVersionMinor();
}

uint32_t DWARFUnit::GetProducerVersionUpdate() {
  return Data().GetProducerVersionUpdate();
}

LanguageType DWARFUnit::LanguageTypeFromDWARF(uint64_t val) {
  // Note: user languages between lo_user and hi_user
  // must be handled explicitly here.
  switch (val) {
  case DW_LANG_Mips_Assembler:
    return eLanguageTypeMipsAssembler;
  case DW_LANG_GOOGLE_RenderScript:
    return eLanguageTypeExtRenderScript;
  default:
    return static_cast<LanguageType>(val);
  }
}

LanguageType DWARFUnit::GetLanguageType() {
  return Data().GetLanguageType();
}

bool DWARFUnit::IsDWARF64() const { return Data().m_is_dwarf64; }

bool DWARFUnit::GetIsOptimized() {
  return Data().GetIsOptimized();
}

SymbolFileDWARFDwo *DWARFUnit::GetDwoSymbolFile() const {
  return Data().m_dwo_symbol_file.get();
}

dw_offset_t DWARFUnit::GetBaseObjOffset() const {
  return Data().m_base_obj_offset;
}

const DWARFDebugInfoEntry *DWARFUnit::GetCompileUnitDIEPtrOnly() {
  return Data().GetCompileUnitDIEPtrOnly();
}

const DWARFDebugInfoEntry *DWARFUnit::DIEPtr() {
  return Data().DIEPtr();
}

void DWARFUnit::Index(NameToDIE &func_basenames,
                             NameToDIE &func_fullnames, NameToDIE &func_methods,
                             NameToDIE &func_selectors,
                             NameToDIE &objc_class_selectors,
                             NameToDIE &globals, NameToDIE &types,
                             NameToDIE &namespaces) {
  assert(!Data().m_dwarf2Data->GetBaseCompileUnit() &&
         "DWARFUnit associated with .dwo or .dwp "
         "should not be indexed directly");

  Log *log(LogChannelDWARF::GetLogIfAll(DWARF_LOG_LOOKUPS));

  if (log) {
    Data().m_dwarf2Data->GetObjectFile()->GetModule()->LogMessage(
        log,
        "DWARFUnit::Index() for compile unit at .debug_info[0x%8.8x]",
        GetOffset());
  }

  const LanguageType cu_language = GetLanguageType();
  DWARFFormValue::FixedFormSizes fixed_form_sizes =
      DWARFFormValue::GetFixedFormSizesForAddressSize(GetAddressByteSize(),
                                                      IsDWARF64());

  IndexPrivate(this, cu_language, fixed_form_sizes, GetOffset(), func_basenames,
               func_fullnames, func_methods, func_selectors,
               objc_class_selectors, globals, types, namespaces);

  SymbolFileDWARFDwo *dwo_symbol_file = GetDwoSymbolFile();
  if (dwo_symbol_file) {
    IndexPrivate(
        dwo_symbol_file->GetCompileUnit(), cu_language, fixed_form_sizes,
        GetOffset(), func_basenames, func_fullnames, func_methods,
        func_selectors, objc_class_selectors, globals, types, namespaces);
  }
}

void DWARFUnit::IndexPrivate(
    DWARFUnit *dwarf_cu, const LanguageType cu_language,
    const DWARFFormValue::FixedFormSizes &fixed_form_sizes,
    const dw_offset_t cu_offset, NameToDIE &func_basenames,
    NameToDIE &func_fullnames, NameToDIE &func_methods,
    NameToDIE &func_selectors, NameToDIE &objc_class_selectors,
    NameToDIE &globals, NameToDIE &types, NameToDIE &namespaces) {
  DWARFDebugInfoEntry::const_iterator pos;
  DWARFDebugInfoEntry::const_iterator begin =
      dwarf_cu->Data().m_die_array.begin();
  DWARFDebugInfoEntry::const_iterator end = dwarf_cu->Data().m_die_array.end();
  for (pos = begin; pos != end; ++pos) {
    const DWARFDebugInfoEntry &die = *pos;

    const dw_tag_t tag = die.Tag();

    switch (tag) {
    case DW_TAG_array_type:
    case DW_TAG_base_type:
    case DW_TAG_class_type:
    case DW_TAG_constant:
    case DW_TAG_enumeration_type:
    case DW_TAG_inlined_subroutine:
    case DW_TAG_namespace:
    case DW_TAG_string_type:
    case DW_TAG_structure_type:
    case DW_TAG_subprogram:
    case DW_TAG_subroutine_type:
    case DW_TAG_typedef:
    case DW_TAG_union_type:
    case DW_TAG_unspecified_type:
    case DW_TAG_variable:
      break;

    default:
      continue;
    }

    DWARFAttributes attributes;
    const char *name = NULL;
    const char *mangled_cstr = NULL;
    bool is_declaration = false;
    // bool is_artificial = false;
    bool has_address = false;
    bool has_location_or_const_value = false;
    bool is_global_or_static_variable = false;

    DWARFFormValue specification_die_form;
    const size_t num_attributes =
        die.GetAttributes(dwarf_cu, fixed_form_sizes, attributes);
    if (num_attributes > 0) {
      for (uint32_t i = 0; i < num_attributes; ++i) {
        dw_attr_t attr = attributes.AttributeAtIndex(i);
        DWARFFormValue form_value;
        switch (attr) {
        case DW_AT_name:
          if (attributes.ExtractFormValueAtIndex(i, form_value))
            name = form_value.AsCString();
          break;

        case DW_AT_declaration:
          if (attributes.ExtractFormValueAtIndex(i, form_value))
            is_declaration = form_value.Unsigned() != 0;
          break;

        //                case DW_AT_artificial:
        //                    if (attributes.ExtractFormValueAtIndex(i,
        //                    form_value))
        //                        is_artificial = form_value.Unsigned() != 0;
        //                    break;

        case DW_AT_MIPS_linkage_name:
        case DW_AT_linkage_name:
          if (attributes.ExtractFormValueAtIndex(i, form_value))
            mangled_cstr = form_value.AsCString();
          break;

        case DW_AT_low_pc:
        case DW_AT_high_pc:
        case DW_AT_ranges:
          has_address = true;
          break;

        case DW_AT_entry_pc:
          has_address = true;
          break;

        case DW_AT_location:
        case DW_AT_const_value:
          has_location_or_const_value = true;
          if (tag == DW_TAG_variable) {
            const DWARFDebugInfoEntry *parent_die = die.GetParent();
            while (parent_die != NULL) {
              switch (parent_die->Tag()) {
              case DW_TAG_subprogram:
              case DW_TAG_lexical_block:
              case DW_TAG_inlined_subroutine:
                // Even if this is a function level static, we don't add it. We
                // could theoretically
                // add these if we wanted to by introspecting into the
                // DW_AT_location and seeing
                // if the location describes a hard coded address, but we dont
                // want the performance
                // penalty of that right now.
                is_global_or_static_variable = false;
                //                              if
                //                              (attributes.ExtractFormValueAtIndex(dwarf2Data,
                //                              i, form_value))
                //                              {
                //                                  // If we have valid block
                //                                  data, then we have location
                //                                  expression bytes
                //                                  // that are fixed (not a
                //                                  location list).
                //                                  const uint8_t *block_data =
                //                                  form_value.BlockData();
                //                                  if (block_data)
                //                                  {
                //                                      uint32_t block_length =
                //                                      form_value.Unsigned();
                //                                      if (block_length == 1 +
                //                                      attributes.CompileUnitAtIndex(i)->GetAddressByteSize())
                //                                      {
                //                                          if (block_data[0] ==
                //                                          DW_OP_addr)
                //                                              add_die = true;
                //                                      }
                //                                  }
                //                              }
                parent_die = NULL; // Terminate the while loop.
                break;

              case DW_TAG_compile_unit:
                is_global_or_static_variable = true;
                parent_die = NULL; // Terminate the while loop.
                break;

              default:
                parent_die =
                    parent_die->GetParent(); // Keep going in the while loop.
                break;
              }
            }
          }
          break;

        case DW_AT_specification:
          if (attributes.ExtractFormValueAtIndex(i, form_value))
            specification_die_form = form_value;
          break;
        }
      }
    }

    switch (tag) {
    case DW_TAG_subprogram:
      if (has_address) {
        if (name) {
          ObjCLanguage::MethodName objc_method(name, true);
          if (objc_method.IsValid(true)) {
            ConstString objc_class_name_with_category(
                objc_method.GetClassNameWithCategory());
            ConstString objc_selector_name(objc_method.GetSelector());
            ConstString objc_fullname_no_category_name(
                objc_method.GetFullNameWithoutCategory(true));
            ConstString objc_class_name_no_category(objc_method.GetClassName());
            func_fullnames.Insert(ConstString(name),
                                  DIERef(cu_offset, die.GetOffset()));
            if (objc_class_name_with_category)
              objc_class_selectors.Insert(objc_class_name_with_category,
                                          DIERef(cu_offset, die.GetOffset()));
            if (objc_class_name_no_category &&
                objc_class_name_no_category != objc_class_name_with_category)
              objc_class_selectors.Insert(objc_class_name_no_category,
                                          DIERef(cu_offset, die.GetOffset()));
            if (objc_selector_name)
              func_selectors.Insert(objc_selector_name,
                                    DIERef(cu_offset, die.GetOffset()));
            if (objc_fullname_no_category_name)
              func_fullnames.Insert(objc_fullname_no_category_name,
                                    DIERef(cu_offset, die.GetOffset()));
          }
          // If we have a mangled name, then the DW_AT_name attribute
          // is usually the method name without the class or any parameters
          const DWARFDebugInfoEntry *parent = die.GetParent();
          bool is_method = false;
          if (parent) {
            dw_tag_t parent_tag = parent->Tag();
            if (parent_tag == DW_TAG_class_type ||
                parent_tag == DW_TAG_structure_type) {
              is_method = true;
            } else {
              if (specification_die_form.IsValid()) {
                DWARFDIE specification_die =
                    dwarf_cu->GetSymbolFileDWARF()->DebugInfo()->GetDIE(
                        DIERef(specification_die_form));
                if (specification_die.GetParent().IsStructOrClass())
                  is_method = true;
              }
            }
          }

          if (is_method)
            func_methods.Insert(ConstString(name),
                                DIERef(cu_offset, die.GetOffset()));
          else
            func_basenames.Insert(ConstString(name),
                                  DIERef(cu_offset, die.GetOffset()));

          if (!is_method && !mangled_cstr && !objc_method.IsValid(true))
            func_fullnames.Insert(ConstString(name),
                                  DIERef(cu_offset, die.GetOffset()));
        }
        if (mangled_cstr) {
          // Make sure our mangled name isn't the same string table entry
          // as our name. If it starts with '_', then it is ok, else compare
          // the string to make sure it isn't the same and we don't end up
          // with duplicate entries
          if (name && name != mangled_cstr &&
              ((mangled_cstr[0] == '_') ||
               (::strcmp(name, mangled_cstr) != 0))) {
            Mangled mangled(ConstString(mangled_cstr), true);
            func_fullnames.Insert(mangled.GetMangledName(),
                                  DIERef(cu_offset, die.GetOffset()));
            ConstString demangled = mangled.GetDemangledName(cu_language);
            if (demangled)
              func_fullnames.Insert(demangled,
                                    DIERef(cu_offset, die.GetOffset()));
          }
        }
      }
      break;

    case DW_TAG_inlined_subroutine:
      if (has_address) {
        if (name)
          func_basenames.Insert(ConstString(name),
                                DIERef(cu_offset, die.GetOffset()));
        if (mangled_cstr) {
          // Make sure our mangled name isn't the same string table entry
          // as our name. If it starts with '_', then it is ok, else compare
          // the string to make sure it isn't the same and we don't end up
          // with duplicate entries
          if (name && name != mangled_cstr &&
              ((mangled_cstr[0] == '_') ||
               (::strcmp(name, mangled_cstr) != 0))) {
            Mangled mangled(ConstString(mangled_cstr), true);
            func_fullnames.Insert(mangled.GetMangledName(),
                                  DIERef(cu_offset, die.GetOffset()));
            ConstString demangled = mangled.GetDemangledName(cu_language);
            if (demangled)
              func_fullnames.Insert(demangled,
                                    DIERef(cu_offset, die.GetOffset()));
          }
        } else
          func_fullnames.Insert(ConstString(name),
                                DIERef(cu_offset, die.GetOffset()));
      }
      break;

    case DW_TAG_array_type:
    case DW_TAG_base_type:
    case DW_TAG_class_type:
    case DW_TAG_constant:
    case DW_TAG_enumeration_type:
    case DW_TAG_string_type:
    case DW_TAG_structure_type:
    case DW_TAG_subroutine_type:
    case DW_TAG_typedef:
    case DW_TAG_union_type:
    case DW_TAG_unspecified_type:
      if (name && !is_declaration)
        types.Insert(ConstString(name), DIERef(cu_offset, die.GetOffset()));
      if (mangled_cstr && !is_declaration)
        types.Insert(ConstString(mangled_cstr),
                     DIERef(cu_offset, die.GetOffset()));
      break;

    case DW_TAG_namespace:
      if (name)
        namespaces.Insert(ConstString(name),
                          DIERef(cu_offset, die.GetOffset()));
      break;

    case DW_TAG_variable:
      if (name && has_location_or_const_value && is_global_or_static_variable) {
        globals.Insert(ConstString(name), DIERef(cu_offset, die.GetOffset()));
        // Be sure to include variables by their mangled and demangled
        // names if they have any since a variable can have a basename
        // "i", a mangled named "_ZN12_GLOBAL__N_11iE" and a demangled
        // mangled name "(anonymous namespace)::i"...

        // Make sure our mangled name isn't the same string table entry
        // as our name. If it starts with '_', then it is ok, else compare
        // the string to make sure it isn't the same and we don't end up
        // with duplicate entries
        if (mangled_cstr && name != mangled_cstr &&
            ((mangled_cstr[0] == '_') || (::strcmp(name, mangled_cstr) != 0))) {
          Mangled mangled(ConstString(mangled_cstr), true);
          globals.Insert(mangled.GetMangledName(),
                         DIERef(cu_offset, die.GetOffset()));
          ConstString demangled = mangled.GetDemangledName(cu_language);
          if (demangled)
            globals.Insert(demangled, DIERef(cu_offset, die.GetOffset()));
        }
      }
      break;

    default:
      continue;
    }
  }
}
